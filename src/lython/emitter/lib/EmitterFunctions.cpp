#include "EmitterCore.h"
#include "EmitterPyOps.h"
#include "EmitterSupport.h"

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
  FunctionSignature sig = types.functionSignature(function);
  if (!diagnoseUnsupportedFunctionSignature(diagnostics, function, sig))
    emitCallableFunction(function, *name, sig, {}, /*isLambda=*/false);
  types.bindSymbol(*name, sig.publicCallable);
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
                callableDefaultValues(builder, callable, sig));
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
  auto restoreGlobalScope = llvm::make_scope_exit([&] {
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
