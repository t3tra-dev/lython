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

namespace lython::emitter {
namespace {

bool diagnoseUnsupportedGeneratorFunction(parser::Diagnostics &diagnostics,
                                          const parser::Node &function,
                                          const FunctionSignature &sig) {
  bool unsupported = false;
  if (sig.generatorAnnotationIncompatible) {
    diagnostics.push_back(parser::Diagnostic{
        parser::Severity::Error, function.range.start,
        "generator function return annotation is incompatible with inferred "
        "Generator or AsyncGenerator contract"});
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
      }
    } else {
      emitCallableFunction(function, *name, sig, {}, /*isLambda=*/false);
    }
  }
  types.bindSymbol(*name, sig.publicCallable);
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

  std::string symbol =
      (llvm::Twine(*name) + "$spec" +
       llvm::Twine(static_cast<unsigned>(generic.specializations.size())))
          .str();
  // Memoize BEFORE emitting the body: monomorphic recursion inside the
  // specialized body must resolve to this same symbol instead of
  // re-specializing.
  generic.specializations[specialized.publicCallable] = symbol;

  // Body annotations spell the type parameters by name (x: T); bind each
  // solved parameter to its ground type for the emission scope, shadowing
  // the generic TypeVar binding the signature pass installed.
  auto scope = types.pushScope();
  for (const auto &binding : bindings)
    types.bindLocalSymbol(binding.first, binding.second);
  emitCallableFunction(*generic.node, symbol, specialized, {},
                       /*isLambda=*/false);
  return std::make_pair(symbol, specialized.publicCallable);
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
  bool savedModuleScope = atModuleScope;
  atModuleScope = false;
  llvm::scope_exit restoreGlobalScope([&] {
    currentGlobalDecls = std::move(savedGlobalDecls);
    atModuleScope = savedModuleScope;
  });
  currentReturnType = sig.resultType;
  currentGeneratorSendType =
      sig.isGeneratorFunction || sig.isAsyncGeneratorFunction
          ? sig.generatorSendType
          : mlir::Type();
  currentFunctionPrefix = symbolName.str();
  types.bindSymbol(symbolName, sig.callable);
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
      values[name] = Value{entry->getArgument(logicalIndex),
                           sig.positionalTypes[logicalIndex]};
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
    types.bindSymbol(capture.name, capture.value.type);
  }

  builder.setInsertionPointToStart(entry);
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
// - NESTED defs keep the zero-argument PROVIDER function called per omitted
//   argument (documented deviation: a nested def re-executes per enclosing
//   call, and there is no per-execution storage to park the value in yet).
mlir::ArrayAttr ModuleEmitter::emitCallableDefaultValues(
    const parser::Node &function, const FunctionSignature &sig,
    llvm::StringRef symbolName) {
  unsigned positionalCount = static_cast<unsigned>(sig.positionalTypes.size());
  llvm::SmallVector<mlir::Attribute, 8> slots(
      positionalCount + sig.kwOnlyTypes.size(), builder.getUnitAttr());

  bool isModuleLevelDef = false;
  if (const auto *moduleBody = ast::nodeList(moduleNode, "body"))
    for (const parser::NodePtr &statement : *moduleBody)
      if (statement.get() == &function) {
        isModuleLevelDef = true;
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
        pendingDefaultCells[&function].push_back(
            PendingDefaultCell{cellName, expr, declared});
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
    if (node && kind && kind.getValue() == "unsupported")
      return emitProvider(node, slot);
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
