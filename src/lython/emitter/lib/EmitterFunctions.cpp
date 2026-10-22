#include "Mangling.h"
#include "AstSynth.h"
#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"
#include "TypeSystemSolver.h"

#include "AstAccess.h"
#include "ClosureAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SaveAndRestore.h"

#include <cstddef>

namespace lython::emitter {
namespace {

bool diagnoseUnsupportedGeneratorFunction(parser::Diagnostics &diagnostics,
                                          const parser::Node &function,
                                          const FunctionSignature &sig) {
  bool unsupported = false;
  if (!sig.generatorAnnotationMismatch.empty()) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             function.range.start,
                                             sig.generatorAnnotationMismatch});
    unsupported = true;
  }
  for (const std::string &reason : sig.generatorAnalysisFailures) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             function.range.start, reason});
    unsupported = true;
  }
  if (sig.asyncGeneratorReturnsValue) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, function.range.start,
        "async generator functions cannot return a value"});
    unsupported = true;
  }
  if (sig.isAsyncGeneratorFunction) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, function.range.start,
        "async generator function lowering is not implemented yet"});
    unsupported = true;
  }
  return unsupported;
}

bool diagnoseUnsupportedFunctionSignature(parser::Diagnostics &diagnostics,
                                          const parser::Node &function,
                                          const FunctionSignature &sig) {
  bool unsupported =
      diagnoseUnsupportedGeneratorFunction(diagnostics, function, sig);
  for (const std::string &name : sig.missingParameterAnnotations) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, function.range.start,
        "function parameter '" + name + "' requires an annotation"});
    unsupported = true;
  }
  for (const std::string &message : sig.invalidParameterAnnotations) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, function.range.start, message});
    unsupported = true;
  }
  for (const std::string &reason : sig.bodyInferenceFailures) {
    diagnostics.push_back(parser::Diagnostic{parser::Severity::Error,
                                             function.range.start, reason});
    unsupported = true;
  }
  return unsupported;
}

} // namespace

void ModuleEmitter::emitFunctionDecl(const parser::Node &function) {
  auto name = ast::string(function, "name");
  if (!name)
    return;
  checkDecorators(function, DecoratorRole::Function);
  FunctionSignature sig = types.functionSignature(function);
  if (!diagnoseUnsupportedFunctionSignature(diagnostics, function, sig)) {
    if (unboundStaticParameterCount(sig.publicCallable) != 0 ||
        forwardsParamSpec(sig)) {
      bool packParameterized = false;
      py::mapPyTypeStructure(
          sig.publicCallable, [&](mlir::Type node) -> std::optional<mlir::Type> {
            if (py::isPyParamSpecType(node) || py::isPyTypeVarTupleType(node))
              packParameterized = true;
            return std::nullopt;
          });
      if (packParameterized) {
        // A pack parameter is a parameter-LIST unknown; one specialization
        // per instantiated arity would need per-call-shape mangling and
        // pack-aware body emission, which the specializer does not do.
        diagnostics.push_back(parser::Diagnostic{
            parser::Severity::Error, function.range.start,
            "generic function '" + std::string(*name) +
                "' uses ParamSpec or TypeVarTuple parameters, which cannot "
                "be specialized yet"});
      } else {
        GenericFunctionInfo &info = genericFunctions[*name];
        info.node = &function;
        info.signature = sig;
        // Specializations inherit the base's symbol, so a generic that shadows
        // a builtin needs the same rename its non-generic sibling gets.
        info.symbolBase = std::string(topLevelFunctionSymbol(*name));
      }
    } else {
      llvm::StringRef symbol = topLevelFunctionSymbol(*name);
      emitCallableFunction(function, symbol, sig, {}, /*isLambda=*/false);
      recordMonomorphicFunction(*name, function, sig, symbol,
                                /*source=*/nullptr);
    }
  }
  types.bindSymbol(*name, sig.publicCallable);
}

void ModuleEmitter::recordMonomorphicFunction(
    llvm::StringRef key, const parser::Node &function,
    const FunctionSignature &sig, llvm::StringRef symbolBase,
    const EmitOptions::SourceModule *source) {
  // Only a plain positional signature can be argument-specialized: the
  // mapping from a call's operands back to the parameters has to be exact,
  // and *args, **kwargs and keyword-only parameters each make it not be.
  // Recording the rest anyway would put the decision in the call site, which
  // is where it would be got wrong once.
  //
  // ⭐ A POSITIONAL DEFAULT DOES NOT. It used to be listed here beside the
  // other three, and the mapping it breaks is only the one from OPERANDS --
  // `specializationArgumentNodes` maps from the declared parameters instead,
  // filling the omitted tail from the defaults, which is exact for a call with
  // no keywords and no `*`. Excluding them left `def go(v: float = 0)` called
  // as `go()` with the declared float ABI and an int default materialised
  // against it, which the lowering reported as "runtime bundle value 0 for
  // '!py.contract<"builtins.float">' has type 'i64'".
  if (sig.varargType || sig.kwargType || !sig.kwOnlyTypes.empty())
    return;
  if (sig.positionalTypes.empty())
    return;
  GenericFunctionInfo &info = monomorphicFunctions[key];
  info.node = &function;
  info.signature = sig;
  info.symbolBase = std::string(symbolBase);
  info.source = source;
}

ModuleEmitter::GenericFunctionInfo *
ModuleEmitter::lookupMonomorphicFunction(llvm::StringRef name) {
  auto found = monomorphicFunctions.find(name);
  if (found != monomorphicFunctions.end())
    return &found->second;
  if (std::optional<std::string> canonical =
          types.lookupCanonicalBinding(name)) {
    found = monomorphicFunctions.find(*canonical);
    if (found != monomorphicFunctions.end())
      return &found->second;
  }
  return nullptr;
}

std::optional<std::pair<std::string, py::CallableType>>
ModuleEmitter::ensureGenericSpecialization(
    const parser::Node &anchor, GenericFunctionInfo &generic,
    py::CallableType target, const ParamSpecCallShape *paramSpecShape) {
  auto name = ast::string(*generic.node, "name");
  auto fail = [&](llvm::StringRef detail) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, anchor.range.start,
        "cannot specialize generic function '" +
            std::string(name.value_or("<lambda>")) + "': " + detail.str()});
    return std::nullopt;
  };
  if (!name)
    return fail("missing function name");
  // Imported generics defer this check to the use site (a module may ship a
  // pack-parameterized function nobody calls); main-module declarations
  // already diagnosed it eagerly in emitFunctionDecl.
  bool packParameterized = false;
  py::mapPyTypeStructure(
      generic.signature.publicCallable,
      [&](mlir::Type node) -> std::optional<mlir::Type> {
        if (py::isPyParamSpecType(node) || py::isPyTypeVarTupleType(node))
          packParameterized = true;
        return std::nullopt;
      });
  if (packParameterized)
    return fail("ParamSpec or TypeVarTuple parameters cannot be "
                "specialized yet");
  if (forwardsParamSpec(generic.signature) && !paramSpecShape)
    return fail("its `*args: P.args` and `**kwargs: P.kwargs` take what each "
                "call passes, so it is specialized at a call and has no value "
                "of its own");

  TypeBindingMap bindings;
  if (!bindSolvedInstantiation(types, generic.signature.publicCallable, target,
                               bindings))
    return fail("the use site does not determine its type arguments");

  FunctionSignature specialized = generic.signature;
  auto substitute = [&](mlir::Type type) {
    return type ? substituteType(types, type, bindings) : type;
  };
  for (mlir::Type &type : specialized.positionalTypes)
    type = substitute(type);
  for (mlir::Type &type : specialized.kwOnlyTypes)
    type = substitute(type);
  specialized.varargType = substitute(specialized.varargType);
  specialized.callableVarargType = substitute(specialized.callableVarargType);
  specialized.kwargType = substitute(specialized.kwargType);
  specialized.resultType = substitute(specialized.resultType);
  specialized.inferredGeneratorType =
      substitute(specialized.inferredGeneratorType);
  specialized.generatorYieldType = substitute(specialized.generatorYieldType);
  specialized.generatorSendType = substitute(specialized.generatorSendType);
  specialized.generatorReturnType =
      substitute(specialized.generatorReturnType);
  // ⭐ P IS WHAT THE CALL PASSED. `Callable[P, R]` is erased to
  // `Callable[..., R]` (any arguments), so nothing in the signature says what
  // `*args: P.args` and `**kwargs: P.kwargs` hold -- the call does, and this
  // body is specialized at its shape: one unnamed positional parameter per
  // argument beyond the declared ones, one keyword-only parameter per keyword
  // none of them takes. The body forwards them one by one
  // (bindParamSpecPacks), so `f(*args, **kwargs)` reaches the callee with the
  // arity and the names the call gave, which a tuple of one and a dict could
  // not say.
  if (paramSpecShape) {
    if (specialized.paramSpecArgs) {
      specialized.positionalDefaults.resize(specialized.positionalTypes.size(),
                                            false);
      specialized.paramSpecArgsType = specialized.varargType;
      specialized.paramSpecPositionalStart =
          static_cast<unsigned>(specialized.positionalTypes.size());
      specialized.paramSpecPositionalCount =
          static_cast<unsigned>(paramSpecShape->positional.size());
      for (mlir::Type member : paramSpecShape->positional) {
        specialized.positionalNames.push_back("");
        specialized.positionalTypes.push_back(member);
        specialized.positionalDefaults.push_back(false);
      }
      specialized.varargType = {};
      specialized.callableVarargType = {};
    }
    if (specialized.paramSpecArgs)
      concretizeParamSpecCallables(*generic.node, specialized,
                                   specialized.paramSpecPositionalStart,
                                   paramSpecShape->positional,
                                   paramSpecShape->keywords,
                                   paramSpecForwarderArguments);
    if (specialized.paramSpecKwargs) {
      specialized.kwOnlyDefaults.resize(specialized.kwOnlyTypes.size(), false);
      specialized.paramSpecKwargsType = specialized.kwargType;
      specialized.paramSpecKwOnlyStart =
          static_cast<unsigned>(specialized.kwOnlyTypes.size());
      specialized.paramSpecKwOnlyCount =
          static_cast<unsigned>(paramSpecShape->keywords.size());
      for (const CallKeywordType &member : paramSpecShape->keywords) {
        specialized.kwOnlyNames.push_back(member.name);
        specialized.kwOnlyTypes.push_back(member.type);
        specialized.kwOnlyDefaults.push_back(false);
      }
      specialized.kwargType = {};
    }
    specialized.paramSpecExpanded = true;
  }
  types.refreshCallable(specialized);
  if (unboundStaticParameterCount(specialized.publicCallable) != 0 ||
      unboundStaticParameterCount(specialized.callable) != 0)
    return fail("the use site leaves type parameters unbound; annotate the "
                "surrounding context");

  auto memoized = generic.specializations.find(specialized.publicCallable);
  if (memoized != generic.specializations.end())
    return std::make_pair(memoized->second, specialized.publicCallable);
  // Divergence backstop for polymorphic recursion: every recursive
  // instantiation at a NEW ground type re-enters here before its body
  // finishes emitting, so an unbounded chain would otherwise recurse
  // forever.
  // ⛔ It counts the instantiations still being emitted -- how deep that
  // re-entry goes -- and not every one there is: a program calling a generic
  // at 33 types, or a ParamSpec forwarder at 33 call shapes, is not
  // recursing, and was refused as if it were.
  if (generic.emittingSpecializations >= 32)
    return fail("too many nested instantiations (polymorphic recursion?)");

  llvm::StringRef symbolBase =
      generic.symbolBase.empty() ? llvm::StringRef(*name) : generic.symbolBase;
  std::string symbol =
      (llvm::Twine(symbolBase) + "$spec" +
       llvm::Twine(static_cast<unsigned>(generic.specializations.size())))
          .str();
  // Memoize BEFORE emitting the body: monomorphic recursion inside the
  // specialized body must resolve to this same symbol instead of
  // re-specializing.
  generic.specializations[specialized.publicCallable] = symbol;

  ++generic.emittingSpecializations;
  auto leave =
      llvm::scope_exit([&] { --generic.emittingSpecializations; });
  auto emitSpecializedBody = [&] {
    // Body annotations spell the type parameters by name (x: T); bind each
    // solved parameter to its ground type for the emission scope, shadowing
    // the generic TypeVar binding the signature pass installed.
    auto scope = types.pushScope();
    for (const auto &binding : bindings) {
      types.bindLocalSymbol(binding.first, binding.second);
      types.bindLocalTypeParameter(binding.first, binding.second);
    }
    emitCallableFunction(*generic.node, symbol, specialized, {},
                         /*isLambda=*/false);
  };
  if (generic.source)
    emitInDefiningModuleScope(*generic.source, emitSpecializedBody);
  else
    emitSpecializedBody();
  return std::make_pair(symbol, specialized.publicCallable);
}

// ⭐ THE CALLABLE P DESCRIBES TAKES THE CALL'S SHAPE. A parameter annotated
// `Callable[P, R]` was erased with the packs, so its argument met
// `Callable[..., R]` -- a target that names no parameter, so a builtin or a
// class handed to it had no forwarder to become (`call_with(print, "hi")`
// reached the lowering as "cannot adapt builtins.function"), and the body's
// `fn(*args)` was checked against nothing. Returns the parameters it typed.
llvm::SmallVector<unsigned, 2> ModuleEmitter::concretizeParamSpecCallables(
    const parser::Node &def, FunctionSignature &sig, unsigned declaredCount,
    llvm::ArrayRef<mlir::Type> packPositional,
    llvm::ArrayRef<CallKeywordType> packKeywords,
    llvm::ArrayRef<bool> needsForwarder) {
  llvm::SmallVector<unsigned, 2> typed;
  const parser::Node *arguments = ast::node(def, "args");
  // ⛔ Only for an argument that needs a forwarder, and only with no keyword
  // in the shape: a program's function is already a callable value, and
  // `def f(a, b=0)` given `b=` by keyword is not a callable with a
  // keyword-only `b` -- typing it as one refused every such call. A
  // forwarder takes positional parameters only.
  if (!sig.paramSpecArgs || !arguments || !packKeywords.empty() ||
      !llvm::is_contained(needsForwarder, true))
    return typed;
  llvm::SmallVector<mlir::Type, 4> keywordTypes;
  llvm::SmallVector<mlir::StringAttr, 4> keywordNames;
  for (const CallKeywordType &member : packKeywords) {
    keywordTypes.push_back(member.type);
    keywordNames.push_back(builder.getStringAttr(member.name));
  }
  auto spellsCallableOfP = [&](const parser::Node *annotation) {
    if (!annotation || annotation->kind != "Subscript")
      return false;
    const parser::Node *slice = ast::node(*annotation, "slice");
    const auto *elements =
        slice && slice->kind == "Tuple" ? ast::nodeList(*slice, "elts")
                                        : nullptr;
    if (!elements || elements->size() != 2 || !elements->front() ||
        elements->front()->kind != "Name")
      return false;
    return ast::nameSpelling(*elements->front()) == *sig.paramSpecArgs;
  };
  for (auto [index, argument] :
       llvm::enumerate(positionalArgumentNodes(*arguments))) {
    if (index >= declaredCount || index >= sig.positionalTypes.size())
      break;
    if (index >= needsForwarder.size() || !needsForwarder[index])
      continue;
    auto erased =
        mlir::dyn_cast_if_present<py::CallableType>(sig.positionalTypes[index]);
    if (!erased || !isGradualCallable(erased) ||
        !spellsCallableOfP(ast::node(*argument, "annotation")))
      continue;
    sig.positionalTypes[index] = py::CallableType::get(
        &context, packPositional, keywordTypes, {}, {},
        erased.getResultTypes(), {}, keywordNames, {}, {});
    typed.push_back(static_cast<unsigned>(index));
  }
  return typed;
}

ModuleEmitter::ParamSpecCallShape
ModuleEmitter::paramSpecCallShape(const FunctionSignature &sig,
                                  const CallOperands &operands) const {
  ParamSpecCallShape shape;
  if (sig.paramSpecArgs)
    for (std::size_t index = sig.positionalTypes.size();
         index < operands.positionalTypes.size(); ++index)
      shape.positional.push_back(
          types.widenLiteral(operands.positionalTypes[index]));
  if (sig.paramSpecKwargs) {
    llvm::StringSet<> named;
    for (auto [index, name] : llvm::enumerate(sig.positionalNames))
      if (index >= sig.positionalOnlyCount)
        named.insert(name);
    for (const std::string &name : sig.kwOnlyNames)
      named.insert(name);
    for (const CallKeywordType &keyword : operands.keywordTypes)
      if (!named.contains(keyword.name))
        shape.keywords.push_back(
            CallKeywordType{keyword.name, types.widenLiteral(keyword.type)});
  }
  return shape;
}

llvm::SmallVector<mlir::Value, 2>
ModuleEmitter::bindParamSpecPacks(const parser::Node &callable,
                                  const FunctionSignature &sig,
                                  mlir::Block *entry) {
  llvm::SmallVector<mlir::Value, 2> held;
  if (sig.varargName && sig.paramSpecArgs) {
    llvm::SmallVector<Value, 4> members;
    for (unsigned index = 0; index < sig.paramSpecPositionalCount; ++index) {
      unsigned position = sig.paramSpecPositionalStart + index;
      members.push_back(
          Value{entry->getArgument(position), sig.positionalTypes[position]});
    }
    // ⛔ An EMPTY pack has no element type to infer; the declared one is the
    // same expectation an inlined method's empty `*args` takes.
    Value packed =
        members.empty() && sig.paramSpecArgsType
            ? emitExprExpected(
                  synth::tuple(std::vector<parser::NodePtr>{}, callable.range)
                      .get(),
                  sig.paramSpecArgsType)
            : emitPack(members);
    if (packed.value &&
        staticPositionalPacks.try_emplace(packed.value, members).second)
      held.push_back(packed.value);
    values[*sig.varargName] = packed;
    types.bindSymbol(*sig.varargName, packed.type);
  }
  if (sig.kwargName && sig.paramSpecKwargs) {
    llvm::SmallVector<std::pair<std::string, Value>, 4> members;
    unsigned firstArgument = static_cast<unsigned>(sig.positionalTypes.size()) +
                             sig.paramSpecKwOnlyStart;
    for (unsigned index = 0; index < sig.paramSpecKwOnlyCount; ++index) {
      unsigned kwIndex = sig.paramSpecKwOnlyStart + index;
      members.push_back({sig.kwOnlyNames[kwIndex],
                         Value{entry->getArgument(firstArgument + index),
                               sig.kwOnlyTypes[kwIndex]}});
    }
    Value packed =
        emitKeywordPackDict(callable, members, sig.paramSpecKwargsType);
    // ⛔ Forwarded by name only while the body never writes it: after
    // `kwargs["x"] = 1` the names the call gave are not its keys any more.
    if (packed.value && keywordPackIsOnlyRead(callable, *sig.kwargName) &&
        staticKeywordPacks.try_emplace(packed.value, members).second)
      held.push_back(packed.value);
    values[*sig.kwargName] = packed;
    types.bindSymbol(*sig.kwargName, packed.type);
  }
  return held;
}

const parser::Node *
ModuleEmitter::defaultExpressionForSlot(const parser::Node &def,
                                        unsigned slot) {
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return nullptr;
  std::size_t positional = 0;
  for (llvm::StringRef field : {"posonlyargs", "args"})
    if (const auto *list = ast::nodeList(*arguments, field))
      positional += list->size();
  if (slot < positional) {
    const auto *defaults = ast::nodeList(*arguments, "defaults");
    if (!defaults || defaults->size() > positional)
      return nullptr;
    std::size_t firstDefault = positional - defaults->size();
    if (slot < firstDefault)
      return nullptr;
    return (*defaults)[slot - firstDefault].get();
  }
  const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults");
  std::size_t index = slot - positional;
  if (!kwDefaults || index >= kwDefaults->size())
    return nullptr;
  return (*kwDefaults)[index].get();
}

std::optional<ModuleEmitter::GenericDefaultCell>
ModuleEmitter::genericDefaultCell(const parser::Node &def, unsigned slot) {
  const GenericFunctionInfo *generic = nullptr;
  for (const auto &entry : genericFunctions)
    if (entry.second.node == &def) {
      generic = &entry.second;
      break;
    }
  if (!generic)
    return std::nullopt;
  const parser::Node *expr = defaultExpressionForSlot(def, slot);
  if (!expr)
    return std::nullopt;
  const FunctionSignature &sig = generic->signature;
  unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
  mlir::Type type;
  if (slot < positionalCount)
    type = sig.positionalTypes[slot];
  else if (slot - positionalCount < sig.kwOnlyTypes.size())
    type = sig.kwOnlyTypes[slot - positionalCount];
  type = type ? types.widenLiteral(type) : type;
  if (!type || unboundStaticParameterCount(type) != 0) {
    type = types.widenLiteral(types.inferExpr(expr));
    if (!type || unboundStaticParameterCount(type) != 0)
      return std::nullopt;
  }
  return GenericDefaultCell{(llvm::Twine("__ly.defaultcell.") +
                             generic->symbolBase + "." + llvm::Twine(slot))
                                .str(),
                            type};
}

std::string ModuleEmitter::genericOmittedDefaultMismatch(
    const parser::Node &def, py::CallableType specialized,
    unsigned positionalGiven, llvm::ArrayRef<CallKeywordType> keywords) {
  if (!specialized)
    return "";
  llvm::ArrayRef<mlir::Type> positionalTypes = specialized.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> kwOnlyTypes = specialized.getKwOnlyTypes();
  const parser::Node *arguments = ast::node(def, "args");
  if (!arguments)
    return "";
  llvm::SmallVector<std::string, 8> names;
  for (llvm::StringRef field : {"posonlyargs", "args", "kwonlyargs"})
    if (const auto *list = ast::nodeList(*arguments, field))
      for (const parser::NodePtr &argument : *list)
        names.push_back(argument ? std::string(ast::nameSpelling(*argument))
                                 : std::string());
  unsigned total =
      static_cast<unsigned>(positionalTypes.size() + kwOnlyTypes.size());
  for (unsigned slot = 0; slot < total && slot < names.size(); ++slot) {
    if (slot < positionalTypes.size() && slot < positionalGiven)
      continue;
    if (llvm::any_of(keywords, [&](const CallKeywordType &keyword) {
          return keyword.name == names[slot];
        }))
      continue;
    const parser::Node *expr = defaultExpressionForSlot(def, slot);
    if (!expr)
      continue;
    mlir::Attribute literal = defaultValueAttr(builder, expr);
    auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(literal);
    auto kind =
        dict ? dict.getAs<mlir::StringAttr>("kind") : mlir::StringAttr();
    if (!kind || kind.getValue() != "unsupported")
      continue;
    mlir::Type parameter = types.widenLiteral(
        slot < positionalTypes.size()
            ? positionalTypes[slot]
            : kwOnlyTypes[slot - positionalTypes.size()]);
    std::optional<GenericDefaultCell> cell = genericDefaultCell(def, slot);
    if (cell && cell->type == parameter)
      continue;
    return "the default of '" + names[slot] +
           "' is one value every call shares, as CPython evaluates it once "
           "at the def" +
           (cell ? ", and it is " + pythonTypeSpelling(cell->type)
                 : std::string()) +
           "; this call leaves it out where the parameter is " +
           pythonTypeSpelling(parameter) + ": pass it";
  }
  return "";
}

const parser::Node *ModuleEmitter::defaultCellOwnerIn(const parser::Node &root,
                                                      const parser::Node &def) {
  const auto *body = ast::nodeList(root, "body");
  if (!body)
    return nullptr;
  for (const parser::NodePtr &statement : *body) {
    if (statement.get() == &def)
      return statement.get();
    if (!statement || statement->kind != "ClassDef")
      continue;
    if (const auto *classBody = ast::nodeList(*statement, "body"))
      for (const parser::NodePtr &member : *classBody)
        if (member.get() == &def)
          return statement.get();
  }
  return nullptr;
}

std::string ModuleEmitter::methodDefaultCellName(const parser::Node &root,
                                                 const parser::Node &owner,
                                                 const parser::Node &def,
                                                 unsigned slot) const {
  std::string module = "__main__";
  for (const EmitOptions::SourceModule &source : options.sourceModules)
    if (source.moduleNode == &root)
      module = source.moduleName;
  return (llvm::Twine("__ly.defaultcell.") + module + "." +
          llvm::StringRef(ast::string(owner, "name").value_or("")) + "." +
          llvm::StringRef(ast::string(def, "name").value_or("")) + "." +
          llvm::Twine(slot))
      .str();
}

// ⭐ A COPY OF A METHOD'S BODY CAN BE EMITTED BEFORE THE METHOD'S OWN. A call
// in a function body is emitted before the module walk reaches the class
// statement, and the cell was registered only by the method's own emission,
// so the copy found none and evaluated the default where it stood:
//
//     class C:
//         def add(self, into: list[int] = []) -> int: ...
//     def through(x: C) -> int: return x.add()
//     print(through(c), through(c))   # printed 1 1; CPython prints 3 4
//
// The cell is registered here instead, under the same name the method's
// emission uses, and written at the class statement either way.
// ⭐ AN IMMUTABLE DEFAULT OF A PROTOCOL PARAMETER IS EVALUATED PER CALL. A
// cell holds what the parameter declares, and a protocol value carries no
// reference count to take when the cell is read: unittest's
// `BaseTestSuite(tests: Iterable[...] = ())` reached the lowering as "module
// global ... has no runtime retain primitive". For a constant, or a tuple of
// them, one evaluation and one per call cannot be told apart.
// ⛔ Not a mutable default, whose sharing between calls is the point: that one
// keeps its cell and the lowering's refusal.
static bool defaultEvaluatedPerCall(mlir::Type declared,
                                    const parser::Node *expression) {
  if (!declared || !mlir::isa<py::ProtocolType>(declared))
    return false;
  std::function<bool(const parser::Node *)> immutable =
      [&](const parser::Node *node) -> bool {
    if (!node)
      return false;
    if (node->kind == "Constant")
      return true;
    if (node->kind != "Tuple")
      return false;
    if (const auto *elements = ast::nodeList(*node, "elts"))
      for (const parser::NodePtr &element : *elements)
        if (!immutable(element.get()))
          return false;
    return true;
  };
  return immutable(expression);
}

std::optional<std::string>
ModuleEmitter::methodDefaultCell(const MethodBinding &method, unsigned slot,
                                 mlir::Type declared) {
  if (!method.method)
    return std::nullopt;
  auto &cells = methodDefaultCells[method.method];
  for (const auto &entry : cells)
    if (entry.first == slot)
      return entry.second;
  const parser::Node *defaultNode =
      defaultExpressionForSlot(*method.method, slot);
  if (!defaultNode || defaultNode->kind == "Constant" || !declared ||
      defaultEvaluatedPerCall(types.widenLiteral(declared), defaultNode))
    return std::nullopt;
  const parser::Node *root = &moduleNode;
  if (const EmitOptions::SourceModule *source =
          sourceModuleForClass(method.definingClass)) {
    if (source->isStub || !source->moduleNode)
      return std::nullopt;
    root = source->moduleNode;
  }
  const parser::Node *owner = defaultCellOwnerIn(*root, *method.method);
  if (!owner || owner == method.method ||
      flushedDefaultCellOwners.contains(owner))
    return std::nullopt;
  // The expression is the def's own node, kept alive by the module's tree.
  parser::NodePtr expression;
  const parser::Node *arguments = ast::node(*method.method, "args");
  for (llvm::StringRef field : {"defaults", "kw_defaults"})
    if (const auto *list = arguments ? ast::nodeList(*arguments, field) : nullptr)
      for (const parser::NodePtr &candidate : *list)
        if (candidate.get() == defaultNode)
          expression = candidate;
  if (!expression)
    return std::nullopt;
  std::string cellName =
      methodDefaultCellName(*root, *owner, *method.method, slot);
  cells.push_back({slot, cellName});
  pendingDefaultCells[owner].push_back(
      PendingDefaultCell{cellName, expression, types.widenLiteral(declared)});
  return cellName;
}

bool ModuleEmitter::statementHasImportedDefaultCells(
    const parser::Node &statement) {
  // A def whose cell is written at its statement: one with a default that is
  // not a literal. A generic one registers nothing pending -- its cell is
  // genericDefaultCell's -- so the defs are what is asked, not the map.
  auto hasCellDefault = [&](const parser::Node &def) {
    for (unsigned slot = 0;; ++slot) {
      const parser::Node *arguments = ast::node(def, "args");
      std::size_t slots = 0;
      if (arguments)
        for (llvm::StringRef field : {"posonlyargs", "args", "kwonlyargs"})
          if (const auto *list = ast::nodeList(*arguments, field))
            slots += list->size();
      if (slot >= slots)
        return false;
      if (const parser::Node *expr = defaultExpressionForSlot(def, slot);
          expr && expr->kind != "Constant")
        return true;
    }
  };
  if (pendingDefaultCells.count(&statement))
    return true;
  return (statement.kind == "FunctionDef" ||
          statement.kind == "AsyncFunctionDef") &&
         hasTypeParameters(statement) && hasCellDefault(statement);
}

// ⭐ AN IMPORTED MODULE'S BODY RUNS TOP TO BOTTOM, as CPython runs it: its
// module globals, the cells of its defs' defaults and its classes'
// attributes are written in the order the statements are written in. They
// were three passes -- every global of every module, then the default
// cells, then every class attribute -- so `_defaultFormatter = Formatter()`
// below `class PercentStyle: default_format = ...` read the class attribute
// before it was written ("referenced before assignment").
// ⛔ Module by module in `sourceModules` order, which is the order the
// passes already took the modules in: only the order WITHIN a module moves.
void ModuleEmitter::emitImportedModuleInitializers() {
  llvm::SmallVector<PendingModuleGlobalInit, 4> globals =
      std::move(importedModuleGlobalInits);
  importedModuleGlobalInits.clear();
  llvm::SmallVector<PendingClassAttrInit, 4> classes =
      std::move(importedClassAttrInits);
  importedClassAttrInits.clear();
  struct Step {
    const parser::Node *anchor;
    const PendingModuleGlobalInit *global = nullptr;
    const PendingClassAttrInit *cls = nullptr;
    const parser::Node *defaults = nullptr;
  };
  for (const EmitOptions::SourceModule &source : options.sourceModules) {
    if (!source.moduleNode || source.isStub)
      continue;
    // Each kind is queued in declaration order; one walk takes them merged.
    llvm::SmallVector<Step, 8> steps;
    for (const PendingModuleGlobalInit &entry : globals)
      if (entry.value && entry.source == &source)
        steps.push_back(Step{entry.value, &entry});
    for (const PendingClassAttrInit &entry : classes)
      if (entry.classDef && entry.source == &source)
        steps.push_back(Step{entry.classDef, nullptr, &entry});
    if (const auto *body = ast::nodeList(*source.moduleNode, "body"))
      for (const parser::NodePtr &statement : *body)
        if (statement && statementHasImportedDefaultCells(*statement))
          steps.push_back(
              Step{statement.get(), nullptr, nullptr, statement.get()});
    if (steps.empty())
      continue;
    llvm::stable_sort(steps, [](const Step &a, const Step &b) {
      return std::make_pair(a.anchor->range.start.line,
                            a.anchor->range.start.column) <
             std::make_pair(b.anchor->range.start.line,
                            b.anchor->range.start.column);
    });
    emitInDefiningModuleScope(source, [&] {
      for (const Step &step : steps) {
        if (step.defaults) {
          emitPendingDefaultCells(*step.defaults);
        } else if (step.cls) {
          emitClassAttrInitializers(*step.cls->classDef,
                                    step.cls->contractName);
          // CPython runs the hook right after the class body it belongs to.
          emitInitSubclassHook(*step.cls->classDef, step.cls->contractName);
        } else if (mlir::Type stored =
                       moduleGlobals.lookup(step.global->globalName)) {
          llvm::SaveAndRestore<std::string> decorating(
              decoratorSubjectName, step.global->decoratorSubject);
          Value initial = emitExprExpected(step.global->value, stored);
          Value coerced = coerceValue(initial, stored, *step.global->value);
          auto op = py::GlobalSetOp::create(
              builder, loc(*step.global->value),
              builder.getStringAttr(step.global->globalName), coerced.value);
          markBoxedModuleGlobal(op);
        }
      }
    });
  }
}

void ModuleEmitter::emitInDefiningModuleScope(
    const EmitOptions::SourceModule &source, llvm::function_ref<void()> body) {
  llvm::SaveAndRestore<std::string> savedSourceName(
      sourceName,
      source.sourceName.empty() ? source.moduleName : source.sourceName);
  llvm::SaveAndRestore<std::string> savedPackageName(activePackageName,
                                                     source.packageName);
  llvm::SaveAndRestore<const parser::Node *> savedSourceModuleNode(
      activeSourceModuleNode, source.moduleNode);
  llvm::SaveAndRestore<const EmitOptions::SourceModule *> savedSourceModule(
      activeSourceModule, &source);
  auto savedLoops = std::move(loopControlContexts);
  loopControlContexts.clear();
  auto savedInlineReturns = std::move(inlineReturnContexts);
  inlineReturnContexts.clear();
  auto savedSupers = std::move(superContexts);
  superContexts.clear();
  // Same reason as in EmitterCalls: this body becomes an LLVM function of its
  // own, which is where its traceback frame comes from.
  auto savedInlineFrames = std::move(inlineFrames);
  inlineFrames.clear();
  llvm::scope_exit restoreContexts([&] {
    loopControlContexts = std::move(savedLoops);
    inlineReturnContexts = std::move(savedInlineReturns);
    superContexts = std::move(savedSupers);
    inlineFrames = std::move(savedInlineFrames);
  });
  std::size_t diagnosticStart = diagnostics.size();
  {
    ImporterModuleScope importerScope(*this);
    TypeSystem::ScopeIsolation isolation = types.isolateScopes();
    auto moduleScope = types.pushScope();
    ++definingScopeSetupDepth;
    bindModuleImportScope(*source.moduleNode, /*diagnoseUnsupported=*/false);
    bindSourceModuleLocals(source.moduleName, *source.moduleNode,
                           source.isStub);
    --definingScopeSetupDepth;
    // Specializations the binding asked for -- a coroutine's signature walks
    // its body, and a generic class it names is specialized there -- were
    // held until the module's names were all bound (see
    // `ensureGenericClassSpecialization`); they go out before the body that
    // may use them.
    if (definingScopeSetupDepth == 0 && genericClassEmissionReady)
      drainGenericClassSpecializations();
    body();
  }
  for (std::size_t index = diagnosticStart; index < diagnostics.size();
       ++index)
    if (diagnostics[index].filename.empty())
      diagnostics[index].filename = sourceName;
}

ModuleEmitter::GenericFunctionInfo *
ModuleEmitter::lookupGenericFunction(llvm::StringRef name) {
  auto found = genericFunctions.find(name);
  if (found != genericFunctions.end())
    return &found->second;
  if (std::optional<std::string> canonical =
          types.lookupCanonicalBinding(name)) {
    found = genericFunctions.find(*canonical);
    if (found != genericFunctions.end())
      return &found->second;
  }
  return nullptr;
}

namespace {

// ⭐ CAN CONTROL REACH THE END OF THIS BODY? The question the function
// epilogue has to answer before it fabricates a `return None`: a function
// annotated `-> str` whose every path raises has no fallthrough to give a
// value to, and one that DOES fall through is a program whose annotation is
// not true on every path.
//
// Both used to end in the same place -- `py.class.upcast None -> str`, which
// the lowering cannot give lanes to -- and the report was "callable return ABI
// expected 2 physical values, but lowering produced 0", which names neither
// the function nor the missing return.
//
// ⛔ CONSERVATIVE IN ONE DIRECTION ONLY: anything not recognised is assumed to
// complete, so the worst this can do is ask for a return that CPython would
// have defaulted.

// Does a `break` in here leave the loop this body belongs to? A nested loop's
// BODY owns its own breaks, so the walk stops there -- but its `else` does
// not: a break in a nested loop's else targets the outer one.
//
// ⛔ Conservative toward "yes". A missed break makes the enclosing
// `while True:` look inescapable, and the emitter then puts a raise at the
// function's end -- which would fire on a program CPython returns None from.
bool bodyBreaksOutOfItsLoop(const std::vector<parser::NodePtr> *body);

bool statementBreaksOutOfItsLoop(const parser::Node &node) {
  llvm::StringRef kind = node.kind;
  if (kind == "Break")
    return true;
  if (kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
      kind == "Lambda" || kind == "ClassDef")
    return false;
  if (kind == "While" || kind == "For" || kind == "AsyncFor")
    return bodyBreaksOutOfItsLoop(ast::nodeList(node, "orelse"));
  for (const parser::Field &field : node.fields) {
    if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
      if (*child && statementBreaksOutOfItsLoop(**child))
        return true;
      continue;
    }
    if (const auto *children =
            std::get_if<std::vector<parser::NodePtr>>(&field.value))
      for (const parser::NodePtr &child : *children)
        if (child && statementBreaksOutOfItsLoop(*child))
          return true;
  }
  return false;
}

bool bodyBreaksOutOfItsLoop(const std::vector<parser::NodePtr> *body) {
  if (!body)
    return false;
  for (const parser::NodePtr &statement : *body)
    if (statement && statementBreaksOutOfItsLoop(*statement))
      return true;
  return false;
}

bool statementCanComplete(const parser::Node &statement) {
  llvm::StringRef kind = statement.kind;
  if (kind == "Raise" || kind == "Return" || kind == "Break" ||
      kind == "Continue")
    return false;
  if (kind == "If") {
    const auto *orelse = ast::nodeList(statement, "orelse");
    if (!orelse || orelse->empty())
      return true;
    return bodyCanComplete(ast::nodeList(statement, "body")) ||
           bodyCanComplete(orelse);
  }
  if (kind == "With" || kind == "AsyncWith")
    return bodyCanComplete(ast::nodeList(statement, "body"));
  // ⭐ `while True:` WITH NO BREAK CANNOT COMPLETE, and it is how a
  // recursive-descent parser's loops are written:
  //
  //     def term(self) -> int:
  //         value = self.factor()
  //         while True:
  //             if self.eat("*"):
  //                 value = value * self.factor()
  //             else:
  //                 return value
  //
  // was refused with "this function can reach its end without returning",
  // which it cannot: the only way out is the return.
  //
  // ⛔ An `else` on such a loop keeps the conservative answer. It is dead code
  // in CPython too, and reading it as unreachable is the direction that puts a
  // raise at a function's end.
  if (kind == "While") {
    const parser::Node *test = ast::node(statement, "test");
    bool alwaysTrue = test && test->kind == "Constant" &&
                      ast::boolean(*test, "value").value_or(false);
    const auto *orelse = ast::nodeList(statement, "orelse");
    if (!alwaysTrue || (orelse && !orelse->empty()))
      return true;
    return bodyBreaksOutOfItsLoop(ast::nodeList(statement, "body"));
  }
  // ⛔ A `match` WITH AN IRREFUTABLE CASE IS EXHAUSTIVE, which is how a
  // dispatch-by-value function ends: `case _:` (or a bare capture) always
  // matches, so if no case body completes, neither does the statement.
  // Treating a match as always completing refused
  //
  //     def classify(v: int) -> str:
  //         match v:
  //             case 0: return "zero"
  //             case _: return "big"
  //
  // with "can reach its end without returning", which it cannot.
  if (kind == "Match") {
    const auto *cases = ast::nodeList(statement, "cases");
    if (!cases || cases->empty())
      return true;
    bool irrefutable = false;
    for (const parser::NodePtr &caseNode : *cases) {
      if (!caseNode)
        return true;
      if (bodyCanComplete(ast::nodeList(*caseNode, "body")))
        return true;
      const parser::Node *pattern = ast::node(*caseNode, "pattern");
      // `case _ as name:` is the same irrefutable capture wearing a second
      // name, and reading only the outer node called it refutable -- so a
      // match whose last case was written that way was reported as able to
      // reach the function's end.
      while (pattern && pattern->kind == "MatchAs" &&
             ast::node(*pattern, "pattern"))
        pattern = ast::node(*pattern, "pattern");
      // A guard can fail, so the case it guards proves nothing.
      if (!ast::node(*caseNode, "guard") && pattern &&
          pattern->kind == "MatchAs" && !ast::node(*pattern, "pattern"))
        irrefutable = true;
    }
    return !irrefutable;
  }
  if (kind == "Try" || kind == "TryStar") {
    // A finally that cannot complete decides for the whole statement,
    // whichever way its body went.
    const auto *finalbody = ast::nodeList(statement, "finalbody");
    if (finalbody && !finalbody->empty() && !bodyCanComplete(finalbody))
      return false;
    const auto *orelse = ast::nodeList(statement, "orelse");
    if (bodyCanComplete(ast::nodeList(statement, "body")) &&
        (!orelse || orelse->empty() || bodyCanComplete(orelse)))
      return true;
    if (const auto *handlers = ast::nodeList(statement, "handlers"))
      for (const parser::NodePtr &handler : *handlers)
        if (handler && bodyCanComplete(ast::nodeList(*handler, "body")))
          return true;
    return false;
  }
  return true;
}

} // namespace

bool bodyCanComplete(const std::vector<parser::NodePtr> *body) {
  if (!body)
    return true;
  for (const parser::NodePtr &statement : *body)
    if (statement && !statementCanComplete(*statement))
      return false;
  return true;
}

void ModuleEmitter::emitCallableFunction(const parser::Node &callable,
                                         llvm::StringRef symbolName,
                                         const FunctionSignature &sig,
                                         llvm::ArrayRef<Capture> captures,
                                         bool isLambda,
                                         unsigned positionalNodeOffset,
                                         mlir::Type preboundTypeObject) {
  if (diagnoseUnsupportedFunctionSignature(diagnostics, callable, sig))
    return;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  llvm::SmallVector<mlir::Type, 8> logicalInputs(sig.positionalTypes.begin(),
                                                 sig.positionalTypes.end());
  logicalInputs.append(sig.kwOnlyTypes.begin(), sig.kwOnlyTypes.end());
  if (sig.varargType)
    logicalInputs.push_back(sig.varargType);
  if (sig.kwargType)
    logicalInputs.push_back(sig.kwargType);
  // ⭐ A CAPTURE'S DECLARED TYPE IS WIDENED, because the value that arrives is.
  // `flag = True` binds `literal<True>` and its runtime bundle is
  // `builtins.bool`, so a nested def capturing it was refused in the lowering:
  // "closure capture 0 has type '!py.contract<"builtins.bool">', expected
  // '!py.literal<True>'". The same capture of an int, a str or a float works,
  // and so does a bool assigned TWICE -- the rebinding is what widened it --
  // which is what says the literal is the accident here.
  //
  // ⛔ Widening every literal rather than the bool: a literal IS assignable to
  // its own contract, so the three kinds that work keep working, and one rule
  // is what stops the next kind from having its own accident.
  auto captureType = [&](const Capture &capture) {
    return types.widenLiteral(capture.value.type);
  };
  for (const Capture &capture : captures)
    logicalInputs.push_back(captureType(capture));

  auto funcType =
      builder.getFunctionType(logicalInputs, mlir::TypeRange{sig.resultType});
  auto func =
      mlir::func::FuncOp::create(builder, loc(callable), symbolName, funcType);
  func.setPrivate();
  func->setAttr("callable_type", mlir::TypeAttr::get(sig.callable));
  // What a `__doc__` read through a value made for this function answers
  // (EmitterDocstrings.cpp); a function the desugars wrote has neither.
  if (auto doc = definitionDocstrings.find(&callable);
      doc != definitionDocstrings.end()) {
    if (doc->second)
      func->setAttr("ly.function.doc", builder.getStringAttr(*doc->second));
    else
      func->setAttr("ly.function.undocumented", builder.getUnitAttr());
  }
  if (auto definition = pythonDefinitionNames.find(&callable);
      definition != pythonDefinitionNames.end()) {
    func->setAttr(py::mangling::kFunctionNameAttr,
                  builder.getStringAttr(definition->second.name));
    func->setAttr(py::mangling::kFunctionQualnameAttr,
                  builder.getStringAttr(definition->second.qualname));
    emittedPythonDefinitions.push_back(EmittedPythonDefinition{
        symbolName.str(), definition->second.module,
        definition->second.qualname});
  }
  if (sig.varargType)
    func->setAttr(kCallableVarargValueTypeAttr,
                  mlir::TypeAttr::get(sig.varargType));
  if (sig.kwargType)
    func->setAttr(kCallableKwargValueTypeAttr,
                  mlir::TypeAttr::get(sig.kwargType));
  func->setAttr("callable_default_values",
                emitCallableDefaultValues(callable, sig, symbolName));
  if (sig.isCoroutineFunction)
    func->setAttr("ly.coroutine", builder.getUnitAttr());
  if (sig.isGeneratorFunction)
    func->setAttr("ly.generator.body_result",
                  mlir::TypeAttr::get(sig.generatorReturnType));
  if (sig.isGeneratorFunction)
    func->setAttr("ly.generator.public_result",
                  mlir::TypeAttr::get(sig.inferredGeneratorType));
  if (!captures.empty()) {
    llvm::SmallVector<std::string, 4> captureNames;
    llvm::SmallVector<mlir::Type, 4> captureTypes;
    for (const Capture &capture : captures) {
      captureNames.push_back(capture.name);
      captureTypes.push_back(captureType(capture));
    }
    func->setAttr("closure_names", stringArray(builder, captureNames));
    func->setAttr("closure_types", typeArray(builder, captureTypes));
  }

  ScopedCallableEmission emissionScope(
      values, currentReturnType, currentFunctionPrefix,
      currentGeneratorSendType, currentGeneratorYieldType, narrowedFromTypes,
      narrowedMemberTypes, types);
  llvm::SaveAndRestore<bool> savedCoroutine(currentFunctionIsCoroutine,
                                            sig.isCoroutineFunction);
  // The forward look for a later read stops at this callable's own suites: a
  // name inside a nested function is a different binding, and the enclosing
  // function's remainder says nothing about it.
  llvm::SaveAndRestore<unsigned> savedSuiteFloor(
      suiteStackFloor, static_cast<unsigned>(suiteStack.size()));

  // ⭐ THIS IS A REAL FUNCTION, whatever it was emitted from. A method body is
  // INLINED at its call site, so a nested def written inside one was emitted
  // with the inliner's state still standing and its `return` branched to the
  // INLINER's continuation block -- "reference to block defined in another
  // region", for an ordinary helper:
  //
  //     class C:
  //         def m(self) -> int:
  //             def h(n: int) -> int:
  //                 return n + 1
  //             return h(1)
  //
  // The loop context goes with it for the same reason: a `break` in this body
  // would have branched out of another region too.
  //
  // ⛔ ONLY the two that produce a cross-region BRANCH. The virtual-dispatch
  // helper (EmitterCalls.cpp) clears the super and inline-frame state as well,
  // and that is right for a SYNTHESIZED function and wrong here: a method is
  // itself emitted through this function, with its super context pushed by the
  // caller, so clearing it made `super()` in every method body "zero-argument
  // super() requires an enclosing class method body" -- 7 goldens.
  auto savedLoops = std::move(loopControlContexts);
  loopControlContexts.clear();
  auto savedInlineReturns = std::move(inlineReturnContexts);
  inlineReturnContexts.clear();
  llvm::scope_exit restoreEmissionContexts([&] {
    loopControlContexts = std::move(savedLoops);
    inlineReturnContexts = std::move(savedInlineReturns);
  });

  mlir::Block *entry = func.addEntryBlock();
  values.clear();
  std::vector<std::string> savedFrameLocals =
      std::exchange(currentFrameLocals, frameLocalOrder(callable));
  llvm::scope_exit restoreFrameLocals(
      [&] { currentFrameLocals = std::move(savedFrameLocals); });
  llvm::StringSet<> savedGlobalDecls = std::move(currentGlobalDecls);
  currentGlobalDecls.clear();
  llvm::StringSet<> savedBoxedLocals = std::move(currentBoxedLocals);
  currentBoxedLocals = isLambda ? llvm::StringSet<>()
                                : nonlocalBoxedNames(callable);
  bool savedModuleScope = atModuleScope;
  atModuleScope = false;
  llvm::scope_exit restoreGlobalScope([&] {
    currentGlobalDecls = std::move(savedGlobalDecls);
    currentBoxedLocals = std::move(savedBoxedLocals);
    atModuleScope = savedModuleScope;
  });
  currentReturnType = sig.resultType;
  currentGeneratorSendType =
      sig.isGeneratorFunction || sig.isAsyncGeneratorFunction
          ? sig.generatorSendType
          : mlir::Type();
  currentGeneratorYieldType =
      (sig.isGeneratorFunction || sig.isAsyncGeneratorFunction) &&
              sig.generatorYieldTypeIsAnnotated
          ? sig.generatorYieldType
          : mlir::Type();
  currentFunctionPrefix = symbolName.str();
  llvm::SaveAndRestore<std::optional<NotImplementedFallback>> savedFallback(
      notImplementedFallback,
      comparisonDunderFallback(ast::string(callable, "name").value_or(""),
                               sig.positionalNames));
  // ⭐ A GENERATOR CALLING ITSELF GETS A GENERATOR, which is `publicCallable`.
  // `callable` is the body's own signature -- for a generator that is the resume
  // result, None -- so a self-call inside the body typed as None and every
  // recursive generator was refused: "static type !py.literal<None> does not
  // provide manifest method '__iter__'", which is the shape of every tree walk.
  types.bindSymbol(symbolName, sig.isGeneratorFunction ||
                                       sig.isAsyncGeneratorFunction
                                   ? sig.publicCallable
                                   : sig.callable);
  // A def renamed away from a builtin spelling is still spelled by its source
  // name inside its own body, so bind the spelling as well: self-recursion has
  // to resolve to the same scoped callable a non-renamed def resolves to,
  // instead of falling out to the module-scope public callable.
  for (const auto &shadowed : shadowedBuiltinSymbols)
    if (shadowed.second == symbolName)
      types.bindSymbol(shadowed.first(), sig.callable);
  std::optional<std::string> preboundTypeObjectName;
  if (const parser::Node *arguments = ast::node(callable, "args")) {
    llvm::SmallVector<const parser::Node *, 8> positional =
        positionalArgumentNodes(*arguments);
    if (preboundTypeObject && positionalNodeOffset > 0 && !positional.empty())
      preboundTypeObjectName =
          std::string(ast::nameSpelling(*positional.front()));
    for (auto [index, argument] : llvm::enumerate(positional)) {
      if (index < positionalNodeOffset)
        continue;
      unsigned logicalIndex =
          static_cast<unsigned>(index) - positionalNodeOffset;
      if (logicalIndex >= sig.positionalTypes.size() ||
          logicalIndex >= entry->getNumArguments())
        break;
      llvm::StringRef name = ast::nameSpelling(*argument);
      Value bound{entry->getArgument(logicalIndex),
                  sig.positionalTypes[logicalIndex]};
      // FIXED. A boxed PARAMETER used to get no cell, so a nested function
      // reading a parameter the body then rebinds captured the entry value:
      //
      //     def make(n: int) -> None:
      //         def get() -> int: return n
      //         n = n * 2
      //         print(get())      # printed 5; CPython prints 10
      //
      // Re-measured 2026-08-14 against CPython 3.14: 10, and the str form and
      // the two-rebind form agree too.
      //
      // ⛔ Kept for the shape of the failed repair, which is still the
      // constraint on anything that wants to create a cell at this point: it
      // was tried HERE, the cell was built, and the nested function then
      // referred to it directly -- "'py.binding.ref' op using value defined
      // outside the region" -- because the capture that routes a cell into a
      // nested body is wired for a cell the BODY created, not one that exists
      // at entry.
      values[name] = bound;
      types.bindSymbol(name, sig.positionalTypes[logicalIndex]);
    }
    if (const auto *kwonly = ast::nodeList(*arguments, "kwonlyargs")) {
      unsigned offset = static_cast<unsigned>(sig.positionalTypes.size());
      for (auto [index, argument] : llvm::enumerate(*kwonly)) {
        if (!argument || index >= sig.kwOnlyTypes.size() ||
            offset + index >= entry->getNumArguments())
          break;
        llvm::StringRef name = ast::nameSpelling(*argument);
        values[name] =
            Value{entry->getArgument(offset + index), sig.kwOnlyTypes[index]};
        types.bindSymbol(name, sig.kwOnlyTypes[index]);
      }
    }
    unsigned variadicOffset = static_cast<unsigned>(sig.positionalTypes.size() +
                                                    sig.kwOnlyTypes.size());
    if (sig.varargType) {
      if (sig.varargName && variadicOffset < entry->getNumArguments()) {
        values[*sig.varargName] =
            Value{entry->getArgument(variadicOffset), sig.varargType};
        types.bindSymbol(*sig.varargName, sig.varargType);
      }
      ++variadicOffset;
    }
    if (sig.kwargType) {
      if (sig.kwargName && variadicOffset < entry->getNumArguments()) {
        values[*sig.kwargName] =
            Value{entry->getArgument(variadicOffset), sig.kwargType};
        types.bindSymbol(*sig.kwargName, sig.kwargType);
      }
    }
  }
  unsigned captureOffset = static_cast<unsigned>(
      sig.positionalTypes.size() + sig.kwOnlyTypes.size() +
      (sig.varargType ? 1 : 0) + (sig.kwargType ? 1 : 0));
  for (auto [index, capture] : llvm::enumerate(captures)) {
    values[capture.name] =
        Value{entry->getArgument(captureOffset + index), capture.value.type};
    // A cell capture binds the NAME to the cell instance, but the name's
    // Python-level type is the content: expressions read through the cell.
    if (isCellContract(capture.value.type)) {
      if (mlir::Type content = cellContentType(capture.value.type))
        types.bindSymbol(capture.name, content);
      else
        types.bindSymbol(capture.name, capture.value.type);
    } else {
      types.bindSymbol(capture.name, capture.value.type);
    }
  }

  builder.setInsertionPointToStart(entry);
  llvm::SmallVector<mlir::Value, 2> paramSpecPacks;
  if (sig.paramSpecExpanded)
    paramSpecPacks = bindParamSpecPacks(callable, sig, entry);
  auto dropParamSpecPacks = llvm::scope_exit([&] {
    for (mlir::Value pack : paramSpecPacks) {
      staticPositionalPacks.erase(pack);
      staticKeywordPacks.erase(pack);
    }
  });
  // ⭐ A boxed PARAMETER gets its cell HERE, where the builder is finally
  // inside the entry block. The assignment path creates a cell on a name's
  // FIRST binding and a parameter is already bound when the body starts, so
  // without this a nested function reading a parameter the body then rebinds
  // captured the entry value:
  //
  //     def make(n: int) -> None:
  //         def get() -> int: return n
  //         n = n * 2
  //         print(get())      # printed 5; CPython prints 10
  //
  // ⛔ NOT at the parameter-binding loop above, which is where it belongs by
  // subject: the builder is not in this function's body yet there, so the
  // cell's ops were emitted at module scope reading the entry block argument
  // -- "'py.binding.ref' op using value defined outside the region".
  for (const auto &boxed : currentBoxedLocals) {
    auto bound = values.find(boxed.getKey());
    if (bound == values.end())
      continue;
    if (!mlir::isa<mlir::BlockArgument>(bound->second.value))
      continue;
    Value cell = emitCellAlloc(callable, bound->second);
    values[boxed.getKey()] = cell;
    types.bindSymbol(boxed.getKey(), cellContentType(cell.type));
  }
  // ⭐ AND AN EMPTY CELL FOR A NAME THE BODY BINDS ONLY LATER. The assignment
  // path makes a cell on a name's FIRST binding, so a nested def that READS
  // the name before that binding had nothing to capture and the name was
  // dropped from the capture walk -- "emit error: unresolved name 'is_odd'"
  // for the mutually recursive pair, which the same two defs at module scope
  // or in a class body have always resolved:
  //
  //     def go(n: int) -> bool:
  //         def is_even(k: int) -> bool:
  //             return True if k == 0 else is_odd(k - 1)
  //         def is_odd(k: int) -> bool:
  //             return False if k == 0 else is_even(k - 1)
  //         return is_even(n)
  //
  // ⛔ Why NOT resolve the sibling to its SYMBOL instead, the way a def's own
  // name resolves inside its body: the pair is circular, so the binding
  // reference for one needs the other's captures and neither can be built
  // first. A cell is what CPython's frame has here, and `tracksBinding` keeps
  // a read before the binding an error rather than a garbage value -- an
  // `UnboundLocalError` where CPython says `NameError` ("cannot access free
  // variable ... in enclosing scope"), which is its base class and catches the
  // same, but not the same name.
  //
  // ⛔ AND THE MUTUAL PAIR LEAKS, because it is a reference CYCLE: the cell
  // holds one function object, whose closure store holds the other, whose
  // store holds the cell. Measured at 889 B per call for a pair that captures
  // a 512-byte string, against 240 B for the same pair capturing nothing. That
  // is this runtime's refcount model and not this fix -- two objects pointing
  // at each other leak 1284 B per iteration, and `xs.append(xs)` leaks 861,
  // both of which CPython collects with a cycle GC this compiler does not
  // have. Refusing the program is not the smaller cost: it compiled to
  // "unresolved name" and nothing ran at all.
  // ⭐ A CLOSED MUTUAL GROUP IS REACHED BY SYMBOL, not through a cell. Members
  // of `closedMutualNestedDefs` capture nothing outside the group, so each one
  // can be named the way a nested def already names ITSELF -- a binding
  // reference with no closure arguments. That is what breaks the cycle the
  // cell otherwise makes (cell -> function object -> closure store -> cell),
  // which this runtime cannot collect and which leaked 400 B per call of the
  // enclosing function.
  //
  // ⛔ The symbols are assigned for the WHOLE group before any member's body
  // is emitted, which is the only ordering that works: the pair is circular,
  // so whichever body goes first has to name a symbol that does not exist yet.
  // Declaring then defining is the same split the sibling-subclass dispatch
  // needed, for the same reason.
  SiblingGroupState enclosingSiblings = std::move(closedSiblingGroup);
  closedSiblingGroup = SiblingGroupState{};
  auto restoreSiblings = llvm::scope_exit(
      [&] { closedSiblingGroup = std::move(enclosingSiblings); });
  if (!isLambda) {
    MutualNestedDefGroup group = mutualNestedDefGroup(callable);
    if (!group.members.empty())
      if (const auto *groupBody = ast::nodeList(callable, "body")) {
        closedSiblingGroup.captures = std::move(group.captures);
        for (const parser::NodePtr &statement : *groupBody) {
          if (!statement || statement->kind != "FunctionDef")
            continue;
          auto memberName = ast::string(*statement, "name");
          if (!memberName || !group.members.contains(*memberName))
            continue;
          closedSiblingGroup.members[*memberName] = ClosedSibling{
              nestedFunctionSymbolName(*memberName, *statement),
              statement.get()};
        }
      }
  }
  if (!isLambda)
    emitForwardBoundCells(callable);
  if (preboundTypeObjectName && preboundTypeObject) {
    mlir::Type classType = types.typeObject(preboundTypeObject);
    auto typeObject = py::TypeObjectOp::create(builder, loc(callable),
                                               classType, preboundTypeObject);
    values[*preboundTypeObjectName] = Value{typeObject.getResult(), classType};
    types.bindSymbol(*preboundTypeObjectName, classType);
  }
  // ⭐ A NESTED DEF CALLING ITSELF. The self-call resolves through the symbol
  // bound above, and for a nested def that symbol is the mangled one, so the
  // SOURCE name was unresolved inside its own body:
  //
  //     def outer(n: int) -> int:
  //         def rec(k: int) -> int:
  //             if k <= 0: return 0
  //             return k + rec(k - 1)     # emit error: unresolved name 'rec'
  //         return rec(n)
  //
  // The reference carries THIS instance's closure arguments, which is what
  // makes it the same closure and not a fresh one: CPython's inner function
  // reads the enclosing frame's cell, and these arguments are that cell's
  // contents. Nothing is materialized when the only use is the call itself.
  //
  // ⛔ Not the enclosing scope's binding, which is what a capture would be:
  // the def statement has not finished executing when the body is emitted, so
  // there is no value there to capture -- which is why the capture walk in
  // emitNestedFunctionDecl finds nothing and drops the name.
  //
  // ⛔ Not for a METHOD: its name is an attribute of the class, and a class
  // scope is not visible from the bodies in it, so the spelling reads the
  // module's binding. `def error(self, msg): return error(msg)` called itself
  // with the arguments one short.
  if (!isLambda && !classBodyDefs.contains(&callable) &&
      !decoratedMethodBodies.contains(&callable))
    if (auto selfName = ast::string(callable, "name");
        selfName && llvm::StringRef(selfName->data(), selfName->size()) !=
                        symbolName &&
        !values.count(llvm::StringRef(selfName->data(), selfName->size()))) {
      llvm::StringRef selfSpelling(selfName->data(), selfName->size());
      llvm::SmallVector<Value, 4> selfCaptures;
      for (auto [index, capture] : llvm::enumerate(captures))
        selfCaptures.push_back(Value{entry->getArgument(captureOffset + index),
                                     capture.value.type});
      mlir::Type selfType =
          sig.isGeneratorFunction || sig.isAsyncGeneratorFunction
              ? sig.publicCallable
              : sig.callable;
      if (selfType && llvm::is_contained(lexicalCaptureNames(callable),
                                         selfSpelling.str())) {
        values[selfSpelling] =
            emitBindingRef(callable, symbolName, selfType, selfCaptures);
        types.bindSymbol(selfSpelling, selfType);
      }
      // ⭐ AND EVERY SIBLING OF THE GROUP, by the same construction. This
      // member's capture arguments ARE the group's list, in the group's order,
      // so they are exactly what a sibling's binding reference needs -- which
      // is the whole reason the list is uniform. Nothing holds a function
      // object, so nothing closes the cycle.
      if (enclosingSiblings.members.count(selfSpelling))
        for (const auto &entry : enclosingSiblings.members) {
          if (entry.getKey() == selfSpelling || !entry.second.node)
            continue;
          FunctionSignature siblingSig =
              types.functionSignature(*entry.second.node);
          mlir::Type siblingType = siblingSig.isGeneratorFunction ||
                                           siblingSig.isAsyncGeneratorFunction
                                       ? siblingSig.publicCallable
                                       : siblingSig.callable;
          if (!siblingType)
            continue;
          values[entry.getKey()] =
              emitBindingRef(callable, entry.second.symbol, siblingType,
                             selfCaptures);
          types.bindSymbol(entry.getKey(), siblingType);
        }
    }
  if (isLambda) {
    Value body = coerceValue(emitExpr(ast::node(callable, "body")),
                             currentReturnType, callable);
    mlir::func::ReturnOp::create(builder, loc(callable), body.value);
  } else {
    // ⭐ A BODY THAT IS ONLY `...` IS A STUB, and a stub whose declared result
    // cannot hold None has no value to fall through with:
    //
    //     class Shape(Protocol):
    //         def area(self) -> int: ...
    //     # this function can reach its end without returning, and its
    //     # declared result builtins.int cannot hold the None a fallthrough
    //     # returns
    //
    // which is the refusal `...` earns once it stops being an unsupported
    // constant. CPython returns None from such a body; the annotation says it
    // cannot, and the program that wrote the stub never means it to be called.
    // So the body IS the raise a hand-written stub spells out, which is the
    // form (`raise NotImplementedError`) this compiler has always accepted.
    //
    // ⛔ Only where None does not fit. `def f() -> None: ...` returns None in
    // CPython and returns None here; raising there would be a deviation for a
    // body whose answer IS representable.
    const auto *bodyStatements = ast::nodeList(callable, "body");
    bool stubBody = isEllipsisStubBody(bodyStatements);
    if (stubBody && currentReturnType &&
        !py::isAssignableTo(types.none(), currentReturnType, module)) {
      parser::NodePtr stub = synth::raiseStmt(
          synth::call(synth::name("NotImplementedError", callable.range),
                      {synth::strConstant(
                          std::string(ast::string(callable, "name")
                                          .value_or("this function")) +
                              " is a stub",
                          callable.range)},
                      callable.range),
          callable.range);
      emitStatement(*stub);
      synthesizedIteratorDefs.push_back(std::move(stub));
    } else {
      emitStatements(bodyStatements);
    }
  }
  if (!insertionBlockTerminated(builder)) {
    auto emitPrimitiveFallbackReturn = [&]() -> bool {
      if (!currentReturnType || py::isPyType(currentReturnType))
        return false;
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(currentReturnType)) {
        auto zero = mlir::arith::ConstantIntOp::create(builder, loc(callable),
                                                       0, integer.getWidth());
        mlir::func::ReturnOp::create(builder, loc(callable), zero.getResult());
        return true;
      }
      return false;
    };
    mlir::Block *currentBlock = builder.getInsertionBlock();
    if (currentBlock && currentBlock != entry &&
        currentBlock->hasNoPredecessors() && emitPrimitiveFallbackReturn())
      return;
    if (currentReturnType && !py::isPyType(currentReturnType)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, callable.range.start,
          "primitive function can fall through without returning a value"});
      if (emitPrimitiveFallbackReturn())
        return;
    }
    // The declared result has to be producible from `None`, because that is
    // what a fallthrough returns. Where it is not, the two cases are the two
    // answers: a body that cannot reach its end has no fallthrough to answer
    // for, and one that can is a program to refuse by name.
    if (!isLambda && currentReturnType &&
        !py::isAssignableTo(types.none(), currentReturnType, module)) {
      if (!bodyCanComplete(ast::nodeList(callable, "body"))) {
        // ⛔ A RAISE AND NOT A VALUE. Nothing reaches here, so any value would
        // be a fiction -- and a fiction that the ABI still has to expand into
        // the declared result's physical lanes. The raise needs none, and if
        // the analysis above is ever wrong the program says so instead of
        // returning a made-up string.
        parser::NodePtr unreachable = synth::raiseStmt(
            synth::call(synth::name("RuntimeError", callable.range),
                        {synth::strConstant(
                            "function reached its end without returning",
                            callable.range)},
                        callable.range),
            callable.range);
        emitStatement(*unreachable);
        synthesizedIteratorDefs.push_back(std::move(unreachable));
        return;
      }
      std::string described;
      llvm::raw_string_ostream stream(described);
      stream << currentReturnType;
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, callable.range.start,
          "this function can reach its end without returning, and its declared "
          "result " +
              described +
              " cannot hold the None a fallthrough returns: return a value on "
              "every path, or widen the annotation to include None"});
      return;
    }
    Value none = emitNone(callable);
    emitFrameExitKeepAlives(callable);
    Value result = coerceValue(none, currentReturnType, callable);
    mlir::func::ReturnOp::create(builder, loc(callable), result.value);
  }
}

// CPython evaluates a nested callable's non-constant defaults when the
// statement -- or the EXPRESSION -- that creates it runs, once per enclosing
// execution and in the enclosing frame. They are evaluated here, where the
// builder still sits in that frame, and threaded in as synthetic captures;
// every omitted-argument call of this instance then shares the one evaluation.
//
// ⭐ THE LAMBDA SPELLING NEEDS IT TOO, and only the def spelling had it. So
// the idiom Python has for capturing a loop variable by value was refused in
// the form everyone writes it:
//
//     fs = []
//     for i in range(3):
//         fs.append(lambda i=i: i * 2)
//     # default expression must not capture enclosing local variable 'i'
//
// while the same three lines written with a `def` compiled and printed
// CPython's [0, 2, 4]. The refusal is the zero-argument PROVIDER fallback's,
// which is called with no closure environment -- correct for a default that
// nothing evaluated here, and never reached once this has.
void ModuleEmitter::evaluateNestedDefaults(
    const parser::Node &function, const FunctionSignature &sig,
    llvm::SmallVectorImpl<Capture> &captures) {
  auto evaluateNestedDefault = [&](const parser::NodePtr &expr,
                                   unsigned slot) {
    if (!expr)
      return;
    mlir::Attribute literal = defaultValueAttr(builder, expr.get());
    auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(literal);
    auto kind = dict ? dict.getAs<mlir::StringAttr>("kind") : mlir::StringAttr();
    if (!kind || kind.getValue() != "unsupported")
      return;
    mlir::Type declared;
    unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
    if (slot < positionalCount)
      declared = types.widenLiteral(sig.positionalTypes[slot]);
    else if (slot - positionalCount < sig.kwOnlyTypes.size())
      declared = types.widenLiteral(sig.kwOnlyTypes[slot - positionalCount]);
    if (!declared)
      return;
    Value value = emitExprExpected(expr.get(), declared);
    Value coerced = coerceValue(value, declared, function);
    std::string captureName =
        (llvm::Twine("__ly.defaultcap.") + llvm::Twine(slot)).str();
    nestedDefaultCaptures[&function].push_back(
        {slot, static_cast<unsigned>(captures.size())});
    captures.push_back(Capture{captureName, coerced});
  };
  const parser::Node *arguments = ast::node(function, "args");
  if (!arguments)
    return;
  unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
  if (const auto *defaults = ast::nodeList(*arguments, "defaults");
      defaults && !defaults->empty()) {
    unsigned firstDefault =
        positionalCount - static_cast<unsigned>(defaults->size());
    for (auto [index, value] : llvm::enumerate(*defaults))
      evaluateNestedDefault(value, firstDefault + static_cast<unsigned>(index));
  }
  if (const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults"))
    for (auto [index, value] : llvm::enumerate(*kwDefaults))
      evaluateNestedDefault(value,
                            positionalCount + static_cast<unsigned>(index));
}

void ModuleEmitter::emitForwardBoundCells(const parser::Node &callable) {
  const auto *body = ast::nodeList(callable, "body");
  if (!body)
    return;
  for (const auto &forward : namesBoundAfterNestedReader(callable)) {
    llvm::StringRef name = forward.getKey();
    if (values.count(name))
      continue;
    // A member of the closed group needs no cell: its siblings name it by
    // symbol, which is what keeps it out of their closure stores.
    if (closedSiblingGroup.members.count(name))
      continue;
    mlir::Type content = inferConditionalLocalType({body}, name);
    // The storability rule the conditional slots end with: a slot for an
    // erased type accepts every write and refuses every read.
    if (!content || py::isPyObjectType(content))
      continue;
    auto unbound = py::UnboundOp::create(builder, loc(callable), content);
    values[name] =
        emitCellAlloc(callable, Value{unbound.getResult(), content},
                      /*tracksBinding=*/true);
    types.bindSymbol(name, content);
  }
}

std::string
ModuleEmitter::nestedFunctionSymbolName(llvm::StringRef name,
                                        const parser::Node &function) {
  return (llvm::Twine(currentFunctionPrefix.empty() ? "__main__"
                                                    : currentFunctionPrefix) +
          "$" + sanitizedSymbolPart(name) + "$" +
          llvm::Twine(++syntheticFunctionCounter) + "$" +
          llvm::Twine(function.range.start.line) + "_" +
          llvm::Twine(function.range.start.column))
      .str();
}

Value ModuleEmitter::emitNestedFunctionDecl(const parser::Node &function) {
  auto name = ast::string(function, "name");
  if (!name)
    return emitNone(function);
  checkDecorators(function, DecoratorRole::Function);

  llvm::SmallVector<Capture, 4> captures;
  auto memberOfGroup = closedSiblingGroup.members.find(*name);
  const bool isGroupMember = memberOfGroup != closedSiblingGroup.members.end() &&
                             memberOfGroup->second.node == &function;
  if (isGroupMember) {
    // ⛔ THE GROUP'S LIST, not this def's own captures. A member names its
    // siblings by symbol and has to pass their captures, and the only list it
    // can be sure of is the one it carries itself -- so every member carries
    // the same one, whether it reads those names or not. A sibling is never in
    // it: capturing one would put the function object back in the closure
    // store, which is the cycle the group exists to avoid.
    for (const std::string &captureName : closedSiblingGroup.captures) {
      auto found = values.find(captureName);
      if (found != values.end())
        captures.push_back(Capture{captureName, found->second});
    }
  } else {
    for (const std::string &captureName : lexicalCaptureNames(function)) {
      auto found = values.find(captureName);
      if (found != values.end())
        captures.push_back(Capture{captureName, found->second});
    }
  }

  FunctionSignature sig = types.functionSignature(function);

  evaluateNestedDefaults(function, sig, captures);

  // A group member's symbol was assigned before any member's body was
  // emitted, so a sibling reading it already named THIS symbol.
  std::string symbolName = isGroupMember
                               ? memberOfGroup->second.symbol
                               : nestedFunctionSymbolName(*name, function);
  if (diagnoseUnsupportedGeneratorFunction(diagnostics, function, sig))
    return emitNone(function);
  emitCallableFunction(function, symbolName, sig, captures, /*isLambda=*/false);
  return emitFunctionObject(function, symbolName, sig.publicCallable, captures);
}

// Defaults that are not literal constants (user-type constructors and other
// expressions):
//
// - MODULE-level defs get CPython's R6 semantics — the expression evaluates
//   ONCE, when __main__ reaches the def statement, into a module-lifetime
//   object-global cell that every omitted-argument call site reads
//   (kind="global"; the evaluation itself is emitted by
//   emitPendingDefaultCells at the skipped declaration's slot in the module
//   body walk, preserving def-execution order).
// - NESTED defs evaluate the expression ONCE when the enclosing execution
//   reaches the def statement, into a synthetic closure capture every
//   omitted-argument call site of that instance reads (kind="capture"; a
//   nested def re-executing per enclosing call re-evaluates, which is the
//   CPython def-statement semantics). The zero-argument PROVIDER fallback
//   remains only for callables outside both paths (lambda defaults).
mlir::ArrayAttr ModuleEmitter::emitCallableDefaultValues(
    const parser::Node &function, const FunctionSignature &sig,
    llvm::StringRef symbolName) {
  unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
  llvm::SmallVector<mlir::Attribute, 8> slots(
      positionalCount + sig.kwOnlyTypes.size(), builder.getUnitAttr());

  // ⭐ A METHOD'S DEFAULT IS EVALUATED ONCE TOO, and it was not:
  //
  //     class Bag:
  //         def add(self, into: list[int] = []) -> int:
  //             into.append(1)
  //             return len(into)
  //     b = Bag(); print(b.add(), b.add())   # printed 1 1; CPython prints 1 2
  //
  //     def make() -> int:
  //         print("eval"); return 1
  //     class Bag:
  //         def get(self, n: int = make()) -> int: ...
  //     Bag().get(); Bag().get()             # printed eval twice
  //
  // The FREE-function spelling of both was already right. The cell path was
  // gated on the def being a direct child of the module body, so a method fell
  // to the zero-argument PROVIDER fallback, which is called at every
  // omitted-argument site -- a fresh list per call, and the side effect again.
  // CPython evaluates a method's defaults once, when the class body executes.
  //
  // ⛔ The owner is the CLASS statement, not the def: the module walk flushes
  // pending cells at the statement it skipped, and for a method that statement
  // is the ClassDef. The note at that call site already said so ("Not
  // ClassDef-exclusive: method defaults registered under a class statement flow
  // through the same cells") -- nothing had ever registered one.
  //
  // ⛔ One level only. A def inside a def keeps the capture path, which is the
  // CPython semantics for it (the inner def re-executes per enclosing call, so
  // its defaults re-evaluate); a class nested in a function is not a
  // module-lifetime cell either.
  //
  // ⭐ AN IMPORTED MODULE'S DEFS TOO. Only the main module was searched, so a
  // def of `lib` fell to the provider and evaluated its default at every call:
  // `lib.f()` with `def f(xs: list[int] = [])` printed 1 1 1 where CPython
  // prints 1 2 3. The module being emitted is the def's own; its cells are
  // written where its other initializers are (emitImportedModuleInitializers).
  const parser::Node &definingModule =
      activeSourceModuleNode ? *activeSourceModuleNode : moduleNode;
  const parser::Node *defaultCellOwner =
      defaultCellOwnerIn(definingModule, function);
  // ⛔ Nor one whose statement already ran: nothing would write the cell.
  // A generic function's is the exception -- its statement writes it whether
  // or not a specialization exists yet, and specializations come later.
  if (defaultCellOwner && flushedDefaultCellOwners.contains(defaultCellOwner) &&
      !(defaultCellOwner == &function && hasTypeParameters(function)))
    defaultCellOwner = nullptr;
  bool isModuleLevelDef = defaultCellOwner != nullptr;
  auto declaredSlotType = [&](unsigned slot) -> mlir::Type {
    if (slot < positionalCount)
      return types.widenLiteral(sig.positionalTypes[slot]);
    unsigned kwIndex = slot - positionalCount;
    if (kwIndex < sig.kwOnlyTypes.size())
      return types.widenLiteral(sig.kwOnlyTypes[kwIndex]);
    return {};
  };

  auto emitProvider = [&](const parser::NodePtr &expr,
                          unsigned slot) -> mlir::Attribute {
    parser::NodePtr lambda = parser::makeNode("Lambda", expr->range);
    parser::NodePtr argumentsNode =
        parser::makeNode("arguments", expr->range);
    parser::addField(*argumentsNode, "posonlyargs",
                     std::vector<parser::NodePtr>{});
    parser::addField(*argumentsNode, "args", std::vector<parser::NodePtr>{});
    parser::addField(*argumentsNode, "kwonlyargs",
                     std::vector<parser::NodePtr>{});
    parser::addField(*argumentsNode, "kw_defaults",
                     std::vector<parser::NodePtr>{});
    parser::addField(*argumentsNode, "defaults",
                     std::vector<parser::NodePtr>{});
    parser::addField(*lambda, "args", argumentsNode);
    parser::addField(*lambda, "body", expr);
    synthesizedDefaultProviders.push_back(lambda);

    // The provider is called without a closure environment, so a default
    // expression must not capture enclosing locals (module-scope classes,
    // functions and constants resolve without captures, like in emitLambda).
    for (const std::string &captureName : lexicalCaptureNames(*lambda)) {
      if (values.find(captureName) == values.end())
        continue;
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr->range.start,
          "default expression must not capture enclosing local variable '" +
              captureName + "'"});
      return builder.getDictionaryAttr({builder.getNamedAttr(
          "kind", builder.getStringAttr("unsupported"))});
    }

    mlir::Type slotDeclared = isModuleLevelDef ? declaredSlotType(slot)
                                               : mlir::Type();
    bool perCall = defaultEvaluatedPerCall(slotDeclared, expr.get());
    if (isModuleLevelDef && !perCall) {
      if (mlir::Type declared = declaredSlotType(slot)) {
        // ⭐ A SPECIALIZATION READS ITS GENERIC'S CELL. Its own cell was
        // registered when the specialization was emitted -- at the first
        // call, which is usually after the module walk had passed the def
        // -- so nothing ever wrote it and an omitted argument read a null
        // object: `def g[T](x: T, xs: list[int] = [1])` then `g(1)` was a
        // segfault. The generic's cell is written at the def statement
        // (emitPendingDefaultCells), once, as CPython evaluates it.
        // ⛔ A specialization whose parameter cannot hold that one value
        // gets no default at all; a call that would read it is refused at
        // the call (genericOmittedDefaultMismatch).
        if (defaultCellOwner == &function && hasTypeParameters(function)) {
          llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
          std::optional<GenericDefaultCell> cell =
              genericDefaultCell(function, slot);
          if (!cell || cell->type != declared) {
            attrs.push_back(builder.getNamedAttr(
                "kind", builder.getStringAttr("unshared")));
            return builder.getDictionaryAttr(attrs);
          }
          attrs.push_back(
              builder.getNamedAttr("kind", builder.getStringAttr("global")));
          attrs.push_back(builder.getNamedAttr(
              "value", builder.getStringAttr(cell->name)));
          return builder.getDictionaryAttr(attrs);
        }
        std::string cellName;
        if (defaultCellOwner != &function) {
          // A method: the cell a call site may already have registered.
          cellName = methodDefaultCellName(definingModule, *defaultCellOwner,
                                           function, slot);
          auto &cells = methodDefaultCells[&function];
          if (llvm::none_of(cells, [&](const auto &entry) {
                return entry.first == slot;
              })) {
            cells.push_back({slot, cellName});
            pendingDefaultCells[defaultCellOwner].push_back(
                PendingDefaultCell{cellName, expr, declared});
          }
        } else {
          cellName = (llvm::Twine("__ly.defaultcell.") + symbolName + "." +
                      llvm::Twine(slot))
                         .str();
          pendingDefaultCells[defaultCellOwner].push_back(
              PendingDefaultCell{cellName, expr, declared});
        }
        llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
        attrs.push_back(
            builder.getNamedAttr("kind", builder.getStringAttr("global")));
        attrs.push_back(builder.getNamedAttr(
            "value", builder.getStringAttr(cellName)));
        return builder.getDictionaryAttr(attrs);
      }
    }

    FunctionSignature providerSig = types.functionSignature(*lambda);
    std::string providerName =
        (llvm::Twine("__ly.default.") + symbolName + "." +
         llvm::Twine(slot))
            .str();
    emitCallableFunction(*lambda, providerName, providerSig, {},
                         /*isLambda=*/true);
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    attrs.push_back(
        builder.getNamedAttr("kind", builder.getStringAttr("provider")));
    attrs.push_back(builder.getNamedAttr(
        "value", builder.getStringAttr(providerName)));
    return builder.getDictionaryAttr(attrs);
  };

  auto attrFor = [&](const parser::NodePtr &node,
                     unsigned slot) -> mlir::Attribute {
    mlir::Attribute literal = defaultValueAttr(builder, node.get());
    auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(literal);
    auto kind = dict ? dict.getAs<mlir::StringAttr>("kind")
                     : mlir::StringAttr();
    if (node && kind && kind.getValue() == "unsupported") {
      // A nested def already evaluated this default at the def statement
      // (emitNestedFunctionDecl) into a synthetic capture; the call site
      // reads it out of the closure evidence instead of re-evaluating.
      if (auto nested = nestedDefaultCaptures.find(&function);
          nested != nestedDefaultCaptures.end()) {
        for (const auto &[capturedSlot, captureIndex] : nested->second) {
          if (capturedSlot != slot)
            continue;
          llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
          attrs.push_back(
              builder.getNamedAttr("kind", builder.getStringAttr("capture")));
          attrs.push_back(builder.getNamedAttr(
              "value", builder.getI64IntegerAttr(captureIndex)));
          return builder.getDictionaryAttr(attrs);
        }
      }
      return emitProvider(node, slot);
    }
    return literal;
  };

  const parser::Node *arguments = ast::node(function, "args");
  const auto *defaults =
      arguments ? ast::nodeList(*arguments, "defaults") : nullptr;
  if (defaults && !defaults->empty()) {
    // The declared positionals end where a ParamSpec's members begin.
    unsigned declaredPositional =
        sig.paramSpecExpanded ? sig.paramSpecPositionalStart : positionalCount;
    unsigned firstDefault = declaredPositional - defaults->size();
    for (auto [index, value] : llvm::enumerate(*defaults))
      if (firstDefault + index < slots.size())
        slots[firstDefault + index] =
            attrFor(value, firstDefault + static_cast<unsigned>(index));
  }
  const auto *kwDefaults =
      arguments ? ast::nodeList(*arguments, "kw_defaults") : nullptr;
  if (kwDefaults) {
    for (auto [index, value] : llvm::enumerate(*kwDefaults)) {
      unsigned slot = positionalCount + static_cast<unsigned>(index);
      if (slot < slots.size())
        slots[slot] = attrFor(value, slot);
    }
  }
  return builder.getArrayAttr(slots);
}

const llvm::StringSet<> &ModuleEmitter::reboundModuleNames() {
  if (!reboundModuleNamesComputed) {
    reboundModuleNamesComputed = true;
    reboundModuleNamesCache = reboundNames(moduleNode);
  }
  return reboundModuleNamesCache;
}

// ⭐ A LAMBDA THAT FORWARDS AN INLINED METHOD'S PACKS FORWARDS THEIR VALUES.
// `addCleanup(self, function, /, *args, **kwargs)` keeps
// `lambda: function(*args, **kwargs)`, and the lambda's body is a function of
// its own: what it captures is a tuple whose arity a one-element pack does not
// carry and a dict whose keys no type carries, so the call in it was refused.
// The call site's values are what the packs hold, so the lambda is emitted as
// a copy whose calls name them -- each bound here, each captured like any
// other name it reads.
// ⛔ A pack the lambda shadows with a parameter of its own is that parameter.
parser::NodePtr
ModuleEmitter::lambdaWithPacksSpelledOut(const parser::Node &lambda) {
  if (staticPositionalPacks.empty() && staticKeywordPacks.empty())
    return nullptr;
  llvm::StringSet<> shadowed;
  if (const parser::Node *arguments = ast::node(lambda, "args"))
    for (llvm::StringRef group :
         {"posonlyargs", "args", "kwonlyargs"})
      if (const auto *list = ast::nodeList(*arguments, group))
        for (const parser::NodePtr &argument : *list)
          if (argument)
            if (auto name = ast::string(*argument, "arg"))
              shadowed.insert(*name);
  auto packed = [&](const parser::Node *node) -> mlir::Value {
    if (!node || node->kind != "Name")
      return {};
    llvm::StringRef name(ast::nameSpelling(*node));
    if (shadowed.contains(name))
      return {};
    auto bound = values.find(name);
    return bound == values.end() ? mlir::Value() : bound->second.value;
  };
  bool forwards = false;
  std::function<void(const parser::Node &)> scan =
      [&](const parser::Node &node) {
        if (node.kind == "Call") {
          if (const auto *args = ast::nodeList(node, "args"))
            for (const parser::NodePtr &arg : *args)
              if (arg && arg->kind == "Starred" &&
                  staticPositionalPacks.count(
                      packed(ast::node(*arg, "value"))))
                forwards = true;
          if (const auto *keywords = ast::nodeList(node, "keywords"))
            for (const parser::NodePtr &keyword : *keywords)
              if (keyword && !ast::string(*keyword, "arg") &&
                  staticKeywordPacks.count(
                      packed(ast::node(*keyword, "value"))))
                forwards = true;
        }
        for (const parser::Field &field : node.fields)
          if (const auto *child = std::get_if<parser::NodePtr>(&field.value)) {
            if (*child)
              scan(**child);
          } else if (const auto *list =
                         std::get_if<std::vector<parser::NodePtr>>(
                             &field.value)) {
            for (const parser::NodePtr &item : *list)
              if (item)
                scan(*item);
          }
      };
  if (const parser::Node *body = ast::node(lambda, "body"))
    scan(*body);
  if (!forwards)
    return nullptr;
  parser::NodePtr copy = synth::cloneTree(lambda);
  auto bindHidden = [&](const Value &value) {
    std::string hidden =
        "__ly_packed" + std::to_string(++syntheticFunctionCounter);
    values[hidden] = value;
    types.bindSymbol(hidden, value.type);
    return hidden;
  };
  std::function<void(parser::Node &)> rewrite = [&](parser::Node &node) {
    if (node.kind == "Call") {
      if (parser::Field *field = parser::findField(node, "args"))
        if (auto *args = std::get_if<std::vector<parser::NodePtr>>(
                &field->value)) {
          std::vector<parser::NodePtr> spelled;
          for (parser::NodePtr &arg : *args) {
            if (arg && arg->kind == "Starred")
              if (auto pack = staticPositionalPacks.find(
                      packed(ast::node(*arg, "value")));
                  pack != staticPositionalPacks.end()) {
                for (const Value &element : pack->second)
                  spelled.push_back(
                      synth::name(bindHidden(element), arg->range));
                continue;
              }
            spelled.push_back(arg);
          }
          *args = std::move(spelled);
        }
      if (parser::Field *field = parser::findField(node, "keywords"))
        if (auto *keywords = std::get_if<std::vector<parser::NodePtr>>(
                &field->value)) {
          std::vector<parser::NodePtr> spelled;
          for (parser::NodePtr &keyword : *keywords) {
            if (keyword && !ast::string(*keyword, "arg"))
              if (auto pack = staticKeywordPacks.find(
                      packed(ast::node(*keyword, "value")));
                  pack != staticKeywordPacks.end()) {
                for (const auto &[name, value] : pack->second)
                  spelled.push_back(synth::keyword(
                      name, synth::name(bindHidden(value), keyword->range),
                      keyword->range));
                continue;
              }
            spelled.push_back(keyword);
          }
          *keywords = std::move(spelled);
        }
    }
    for (parser::Field &field : node.fields)
      if (auto *child = std::get_if<parser::NodePtr>(&field.value)) {
        if (*child)
          rewrite(**child);
      } else if (auto *list =
                     std::get_if<std::vector<parser::NodePtr>>(&field.value)) {
        for (parser::NodePtr &item : *list)
          if (item)
            rewrite(*item);
      }
  };
  if (parser::Field *body = parser::findField(*copy, "body"))
    if (auto *node = std::get_if<parser::NodePtr>(&body->value); node && *node)
      rewrite(**node);
  synthesizedIteratorDefs.push_back(copy);
  return copy;
}

Value ModuleEmitter::emitLambda(const parser::Node &expr,
                                py::CallableType expected) {
  if (parser::NodePtr spelled = lambdaWithPacksSpelledOut(expr))
    return emitLambda(*spelled, expected);
  llvm::SmallVector<Capture, 4> captures;
  // ⭐ A capture is the value the name holds NOW, and at module scope a
  // rebound name goes stale in it. CPython reads the binding at CALL time:
  //
  //     x = 1
  //     f = lambda: x
  //     x = 2
  //     print(f())                        # printed 1; CPython prints 2
  //
  //     fs = []
  //     for i in range(3):
  //         fs.append(lambda: i)
  //     print([f() for f in fs])          # printed [0, 1, 2]; CPython [2, 2, 2]
  //
  // ⛔ Why REFUSED and not repaired: the def spelling of the same read is
  // already refused ("unresolved name 'x'") because a module name bound more
  // than once gets no cell -- `x = 1` then `x = "a"` is legal Python that one
  // cell cannot hold, which is why the bound-once restriction exists in
  // EmitterGlobals. The lambda took the value instead and was the only
  // spelling that mis-executed silently. Inside a FUNCTION the same read is
  // repaired rather than refused: nonlocalBoxedNames boxes a rebound local
  // that any nested callable reads, lambdas included.
  llvm::StringSet<> moduleRebound;
  if (atModuleScope)
    moduleRebound = reboundModuleNames();
  for (const std::string &captureName : lexicalCaptureNames(expr)) {
    auto found = values.find(captureName);
    if (found == values.end())
      continue;
    // A cell capture is by REFERENCE, so it reads the binding when it runs
    // and there is nothing to go stale: a loop target read by a callable in
    // the loop body is given one before the loop (emitFor).
    if (moduleRebound.contains(captureName) &&
        !isCellContract(found->second.type)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda reads module-level name '" + captureName +
              "', which is rebound after this point"});
      continue;
    }
    captures.push_back(Capture{captureName, found->second});
  }

  FunctionSignature sig = types.functionSignature(expr, std::nullopt, expected);
  if (expected) {
    if (sig.positionalTypes.size() != expected.getPositionalTypes().size() ||
        sig.kwOnlyTypes.size() != expected.getKwOnlyTypes().size() ||
        expected.hasVararg() || expected.hasKwarg()) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda does not match its Callable annotation shape"});
    }
    bool unresolvedUnknown =
        hasUnexpectedObjectTop(sig.callable, expected, types);
    if (unresolvedUnknown) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda Callable annotation does not resolve all Unknown types"});
    }
    if (!unresolvedUnknown &&
        !py::isAssignableTo(widenInferredLiterals(sig.callable, types),
                            expected)) {
      diagnostics.push_back(parser::Diagnostic{
          parser::Severity::Error, expr.range.start,
          "lambda body is not compatible with its Callable annotation"});
    }
  } else if (containsObjectTop(sig.callable, types)) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, expr.range.start,
        "lambda requires a Callable annotation because its type contains "
        "unresolved Unknown"});
  }

  std::string symbolName =
      (llvm::Twine(currentFunctionPrefix.empty() ? "__main__"
                                                 : currentFunctionPrefix) +
       "$lambda$" + llvm::Twine(++syntheticFunctionCounter) + "$" +
       llvm::Twine(expr.range.start.line) + "_" +
       llvm::Twine(expr.range.start.column))
          .str();
  evaluateNestedDefaults(expr, sig, captures);
  emitCallableFunction(expr, symbolName, sig, captures, /*isLambda=*/true);
  return emitFunctionObject(expr, symbolName, sig.callable, captures);
}

} // namespace lython::emitter
