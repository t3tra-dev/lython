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
    if (unboundStaticParameterCount(sig.publicCallable) != 0) {
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
  // and defaults, *args, **kwargs and keyword-only parameters each make it
  // not be. Recording the rest anyway would put the decision in the call
  // site, which is where it would be got wrong once.
  if (sig.varargType || sig.kwargType || !sig.kwOnlyTypes.empty())
    return;
  for (bool hasDefault : sig.positionalDefaults)
    if (hasDefault)
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
ModuleEmitter::ensureGenericSpecialization(const parser::Node &anchor,
                                           GenericFunctionInfo &generic,
                                           py::CallableType target) {
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

  TypeBindingMap bindings;
  if (!target ||
      !bindExpectedType(types, generic.signature.publicCallable, target,
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
  if (generic.specializations.size() >= 32)
    return fail("too many distinct instantiations (polymorphic recursion?)");

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

void ModuleEmitter::emitInDefiningModuleScope(
    const EmitOptions::SourceModule &source, llvm::function_ref<void()> body) {
  llvm::SaveAndRestore<std::string> savedSourceName(
      sourceName,
      source.sourceName.empty() ? source.moduleName : source.sourceName);
  llvm::SaveAndRestore<std::string> savedPackageName(activePackageName,
                                                     source.packageName);
  auto savedLoops = std::move(loopControlContexts);
  loopControlContexts.clear();
  auto savedInlineReturns = std::move(inlineReturnContexts);
  inlineReturnContexts.clear();
  auto savedSupers = std::move(superContexts);
  superContexts.clear();
  llvm::scope_exit restoreContexts([&] {
    loopControlContexts = std::move(savedLoops);
    inlineReturnContexts = std::move(savedInlineReturns);
    superContexts = std::move(savedSupers);
  });
  std::size_t diagnosticStart = diagnostics.size();
  {
    ImporterModuleScope importerScope(*this);
    TypeSystem::ScopeIsolation isolation = types.isolateScopes();
    auto moduleScope = types.pushScope();
    bindModuleImportScope(*source.moduleNode, /*diagnoseUnsupported=*/false);
    bindSourceModuleLocals(source.moduleName, *source.moduleNode,
                           source.isStub);
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
  for (const Capture &capture : captures)
    logicalInputs.push_back(capture.value.type);

  auto funcType =
      builder.getFunctionType(logicalInputs, mlir::TypeRange{sig.resultType});
  auto func =
      mlir::func::FuncOp::create(builder, loc(callable), symbolName, funcType);
  func.setPrivate();
  func->setAttr("callable_type", mlir::TypeAttr::get(sig.callable));
  if (sig.varargType)
    func->setAttr(kCallableVarargValueTypeAttr,
                  mlir::TypeAttr::get(sig.varargType));
  if (sig.kwargType)
    func->setAttr(kCallableKwargValueTypeAttr,
                  mlir::TypeAttr::get(sig.kwargType));
  func->setAttr("callable_default_values",
                emitCallableDefaultValues(callable, sig, symbolName));
  if (callable.kind == "AsyncFunctionDef")
    func->setAttr("ly.async.body_result", mlir::TypeAttr::get(sig.resultType));
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
      captureTypes.push_back(capture.value.type);
    }
    func->setAttr("closure_names", stringArray(builder, captureNames));
    func->setAttr("closure_types", typeArray(builder, captureTypes));
  }

  ScopedCallableEmission emissionScope(values, currentReturnType,
                                       currentFunctionPrefix,
                                       currentGeneratorSendType, types);

  mlir::Block *entry = func.addEntryBlock();
  values.clear();
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
  currentFunctionPrefix = symbolName.str();
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
  if (preboundTypeObjectName && preboundTypeObject) {
    mlir::Type classType = types.typeObject(preboundTypeObject);
    auto typeObject = py::TypeObjectOp::create(builder, loc(callable),
                                               classType, preboundTypeObject);
    values[*preboundTypeObjectName] = Value{typeObject.getResult(), classType};
    types.bindSymbol(*preboundTypeObjectName, classType);
  }
  if (isLambda) {
    Value body = coerceValue(emitExpr(ast::node(callable, "body")),
                             currentReturnType, callable);
    mlir::func::ReturnOp::create(builder, loc(callable), body.value);
  } else {
    emitStatements(ast::nodeList(callable, "body"));
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
    Value none = emitNone(callable);
    Value result = coerceValue(none, currentReturnType, callable);
    mlir::func::ReturnOp::create(builder, loc(callable), result.value);
  }
}

Value ModuleEmitter::emitNestedFunctionDecl(const parser::Node &function) {
  auto name = ast::string(function, "name");
  if (!name)
    return emitNone(function);
  checkDecorators(function, DecoratorRole::Function);

  llvm::SmallVector<Capture, 4> captures;
  for (const std::string &captureName : lexicalCaptureNames(function)) {
    auto found = values.find(captureName);
    if (found != values.end())
      captures.push_back(Capture{captureName, found->second});
  }

  FunctionSignature sig = types.functionSignature(function);

  // CPython evaluates a def statement's non-constant defaults when the def
  // executes — for a nested def that is once per ENCLOSING execution, in the
  // enclosing frame. Evaluate them here (the builder still sits in the
  // enclosing body) and thread each value in as a synthetic capture; every
  // omitted-argument call of this instance then shares the one evaluation.
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
  if (const parser::Node *arguments = ast::node(function, "args")) {
    unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
    if (const auto *defaults = ast::nodeList(*arguments, "defaults");
        defaults && !defaults->empty()) {
      unsigned firstDefault =
          positionalCount - static_cast<unsigned>(defaults->size());
      for (auto [index, value] : llvm::enumerate(*defaults))
        evaluateNestedDefault(value,
                              firstDefault + static_cast<unsigned>(index));
    }
    if (const auto *kwDefaults = ast::nodeList(*arguments, "kw_defaults"))
      for (auto [index, value] : llvm::enumerate(*kwDefaults))
        evaluateNestedDefault(value,
                              positionalCount + static_cast<unsigned>(index));
  }

  std::string symbolName =
      (llvm::Twine(currentFunctionPrefix.empty() ? "__main__"
                                                 : currentFunctionPrefix) +
       "$" + sanitizedSymbolPart(*name) + "$" +
       llvm::Twine(++syntheticFunctionCounter) + "$" +
       llvm::Twine(function.range.start.line) + "_" +
       llvm::Twine(function.range.start.column))
          .str();
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
  bool isModuleLevelDef = false;
  const parser::Node *defaultCellOwner = nullptr;
  if (const auto *moduleBody = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *moduleBody) {
      if (statement.get() == &function) {
        isModuleLevelDef = true;
        defaultCellOwner = statement.get();
        break;
      }
      if (!statement || statement->kind != "ClassDef")
        continue;
      if (const auto *classBody = ast::nodeList(*statement, "body"))
        for (const parser::NodePtr &member : *classBody)
          if (member.get() == &function) {
            isModuleLevelDef = true;
            defaultCellOwner = statement.get();
            break;
          }
      if (isModuleLevelDef)
        break;
    }
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

    if (isModuleLevelDef) {
      if (mlir::Type declared = declaredSlotType(slot)) {
        std::string cellName = (llvm::Twine("__ly.defaultcell.") + symbolName +
                                "." + llvm::Twine(slot))
                                   .str();
        pendingDefaultCells[defaultCellOwner].push_back(
            PendingDefaultCell{cellName, expr, declared});
        if (defaultCellOwner != &function)
          methodDefaultCells[&function].push_back({slot, cellName});
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
    unsigned firstDefault = positionalCount - defaults->size();
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

Value ModuleEmitter::emitLambda(const parser::Node &expr,
                                py::CallableType expected) {
  llvm::SmallVector<Capture, 4> captures;
  for (const std::string &captureName : lexicalCaptureNames(expr)) {
    auto found = values.find(captureName);
    if (found != values.end())
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
  emitCallableFunction(expr, symbolName, sig, captures, /*isLambda=*/true);
  return emitFunctionObject(expr, symbolName, sig.callable, captures);
}

} // namespace lython::emitter
