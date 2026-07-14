#pragma once

#include "Ast.h"
#include "PyDialectTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>

namespace lython::emitter {

struct FunctionSignature {
  py::CallableType callable;
  py::CallableType publicCallable;
  llvm::SmallVector<std::string, 8> positionalNames;
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<std::string, 4> kwOnlyNames;
  llvm::SmallVector<mlir::Type, 4> kwOnlyTypes;
  llvm::SmallVector<bool, 8> positionalDefaults;
  llvm::SmallVector<bool, 4> kwOnlyDefaults;
  llvm::SmallVector<std::string, 4> missingParameterAnnotations;
  llvm::SmallVector<std::string, 4> invalidParameterAnnotations;
  llvm::SmallVector<std::string, 4> bodyInferenceFailures;
  // Runtime-local type of the `args` variable. For `*args: Unpack[Ts]` this is
  // a tuple object; the Callable contract tail is kept separately below.
  mlir::Type varargType;
  mlir::Type callableVarargType;
  mlir::Type kwargType;
  std::optional<std::string> varargName;
  std::optional<std::string> kwargName;
  unsigned positionalOnlyCount = 0;
  mlir::Type resultType;
  mlir::Type publicResultType;
  bool isAsyncFunction = false;
  bool isGeneratorFunction = false;
  bool isAsyncGeneratorFunction = false;
  bool generatorAnnotationIncompatible = false;
  bool asyncGeneratorReturnsValue = false;
  llvm::SmallVector<std::string, 4> generatorAnalysisFailures;
  mlir::Type inferredGeneratorType;
  mlir::Type generatorYieldType;
  mlir::Type generatorSendType;
  mlir::Type generatorReturnType;
};

struct CallKeywordType {
  std::string name;
  mlir::Type type;
};

struct CallInferenceEvidence {
  mlir::Type callableContract;
  std::string methodName;
  std::optional<std::string> receiverManifestClass;
};

struct CallInferenceResult {
  mlir::Type resultType;
  CallInferenceEvidence evidence;
  bool resolved = false;
  std::string failureReason;

  explicit operator bool() const {
    return resolved && static_cast<bool>(resultType);
  }
};

// Strict expression-inference context: names resolve against the enclosing
// function's local callables first, and inference failures propagate as a
// null type with a recorded reason instead of falling back to object().
struct ExprInferenceContext {
  const llvm::StringMap<mlir::Type> &localCallables;
  llvm::SmallVectorImpl<std::string> *failureReasons = nullptr;
  // Locals bound so far by a body walk (assignments, loop targets); shadows
  // the symbol table without mutating it.
  const llvm::StringMap<mlir::Type> *localSymbols = nullptr;
  // Non-strict contexts keep the object() fallbacks but still see
  // localCallables/localSymbols.
  bool strict = true;
};

struct AwaitInferenceResult {
  mlir::Type resultType;
  mlir::Type awaitContract;
  bool resolved = false;
  std::string failureReason;

  explicit operator bool() const {
    return resolved && static_cast<bool>(resultType);
  }
};

struct YieldFromInferenceResult {
  mlir::Type elementType;
  mlir::Type completionType;
  mlir::Type protocolContract;
  bool resolved = false;
  std::string failureReason;

  explicit operator bool() const {
    return resolved && static_cast<bool>(elementType) &&
           static_cast<bool>(completionType) &&
           static_cast<bool>(protocolContract);
  }
};

struct AsyncIterationInferenceResult {
  mlir::Type iteratorType;
  mlir::Type nextAwaitableType;
  mlir::Type itemType;
  CallInferenceResult aiter;
  CallInferenceResult anext;
  AwaitInferenceResult awaitNext;
  bool resolved = false;
  std::string failureReason;

  explicit operator bool() const {
    return resolved && static_cast<bool>(iteratorType) &&
           static_cast<bool>(itemType);
  }
};

struct AsyncContextMethodInferenceResult {
  mlir::Type awaitableType;
  mlir::Type resultType;
  CallInferenceResult method;
  AwaitInferenceResult awaitResult;
  bool resolved = false;
  std::string failureReason;

  explicit operator bool() const {
    return resolved && static_cast<bool>(awaitableType) &&
           static_cast<bool>(resultType);
  }
};

class AlgorithmM {
public:
  class Scope {
  public:
    Scope() = default;
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    Scope(Scope &&other) noexcept;
    Scope &operator=(Scope &&other) noexcept;
    ~Scope();

  private:
    friend class AlgorithmM;
    explicit Scope(const AlgorithmM &owner) : owner(&owner) {}
    void reset();

    const AlgorithmM *owner = nullptr;
  };

  explicit AlgorithmM(mlir::MLIRContext &context);

  mlir::MLIRContext &getContext() const { return context; }
  void seedBuiltins();

  mlir::Type object() const;
  mlir::Type any() const;
  mlir::Type none() const;
  mlir::Type boolType() const;
  mlir::Type intType() const;
  mlir::Type strType() const;
  mlir::Type floatType() const;
  mlir::Type contract(llvm::StringRef name,
                      mlir::ArrayRef<mlir::Type> arguments = {}) const;
  mlir::Type protocol(llvm::StringRef name,
                      mlir::ArrayRef<mlir::Type> arguments = {}) const;
  mlir::Type literal(llvm::StringRef spelling) const;
  mlir::Type typeObject(mlir::Type instanceType) const;
  mlir::Type tupleOf(mlir::Type elementType) const;
  mlir::Type listOf(mlir::Type elementType) const;
  mlir::Type dictOf(mlir::Type keyType, mlir::Type valueType) const;
  mlir::Type iteratorOf(mlir::Type elementType) const;
  mlir::Type coroutineOf(mlir::Type resultType) const;
  // Manifest-driven contract refinement on field assignment
  // (ly.typing.field_param_bindings): the refined receiver type when
  // `receiver.field = value` binds one of the receiver class's type
  // parameters, nullopt otherwise. Pure kernel rule -- which classes/fields
  // participate is declared entirely in the module manifests.
  std::optional<mlir::Type>
  fieldAssignmentRefinement(mlir::Type receiverType, llvm::StringRef fieldName,
                            mlir::Type valueType) const;

  // Target triple for platform-constant typing (sys.platform / os.name /
  // platform.system() infer as string literals of THIS target).
  void setTargetTriple(std::string triple) { targetTriple = std::move(triple); }

  Scope pushScope() const;
  void bindLocalSymbol(llvm::StringRef name, mlir::Type type) const;
  void bindSymbol(llvm::StringRef name, mlir::Type type);
  void bindCanonicalSymbol(llvm::StringRef name, llvm::StringRef canonical,
                           mlir::Type type);
  std::optional<mlir::Type> lookupSymbol(llvm::StringRef name) const;
  std::optional<std::string> lookupCanonicalBinding(llvm::StringRef name) const;
  void bindClass(llvm::StringRef name, mlir::Type instanceType);
  std::optional<mlir::Type> lookupClass(llvm::StringRef name) const;
  bool bindImportedModule(llvm::StringRef module, llvm::StringRef localName);
  bool bindImportedName(llvm::StringRef module, llvm::StringRef exportedName,
                        llvm::StringRef localName);

  mlir::Type annotationType(const parser::Node *node) const;
  mlir::Type inferExpr(const parser::Node *node) const;
  mlir::Type inferExpr(const parser::Node *node,
                       const ExprInferenceContext &ctx) const;
  CallInferenceResult
  inferCallWithEvidence(mlir::Type calleeType,
                        mlir::ArrayRef<mlir::Type> positional,
                        mlir::ArrayRef<CallKeywordType> keywords) const;
  CallInferenceResult inferMethodCallWithEvidence(
      mlir::Type receiverType, llvm::StringRef methodName,
      mlir::ArrayRef<mlir::Type> positional,
      mlir::ArrayRef<CallKeywordType> keywords = {}) const;
  // Manifest fact (`ly.typing.structural_mutators`): the method structurally
  // mutates the receiver, so its call rebinds the receiver local through an
  // extra receiver-typed call result.
  bool isStructuralMutatorMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const;
  // Ordered `__match_args__` attribute names for positional class patterns on
  // the receiver's class; nullopt when the class declares none.
  std::optional<std::vector<std::string>>
  classMatchArgs(mlir::Type receiverType) const;
  AwaitInferenceResult inferAwaitWithEvidence(mlir::Type awaitableType) const;
  YieldFromInferenceResult
  inferYieldFromWithEvidence(mlir::Type sourceType) const;
  AsyncIterationInferenceResult
  inferAsyncIterationWithEvidence(mlir::Type iterableType) const;
  AsyncContextMethodInferenceResult
  inferAsyncContextEnterWithEvidence(mlir::Type managerType) const;
  AsyncContextMethodInferenceResult inferAsyncContextExitWithEvidence(
      mlir::Type managerType, mlir::ArrayRef<mlir::Type> exceptionTypes) const;
  mlir::Type inferCall(mlir::Type calleeType,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords) const;
  mlir::Type
  inferClassInstantiation(mlir::Type instanceType,
                          mlir::ArrayRef<mlir::Type> positional,
                          mlir::ArrayRef<CallKeywordType> keywords) const;
  mlir::Type join(mlir::ArrayRef<mlir::Type> types) const;
  mlir::Type widenLiteral(mlir::Type type) const;

  FunctionSignature
  functionSignature(const parser::Node &function,
                    std::optional<llvm::StringRef> selfName = std::nullopt,
                    py::CallableType expectedCallable = {}) const;
  void refreshCallable(FunctionSignature &sig) const;

private:
  mlir::Type inferExprImpl(const parser::Node *node,
                           const ExprInferenceContext *ctx) const;
  void popScope() const;
  void bindAnnotationAlias(llvm::StringRef name, llvm::StringRef target);
  std::string resolveAnnotationName(llvm::StringRef name) const;

  mlir::MLIRContext &context;
  std::string targetTriple;
  llvm::StringMap<mlir::Type> symbols;
  llvm::StringMap<mlir::Type> classes;
  llvm::StringMap<std::string> canonicalBindings;
  llvm::StringMap<std::string> annotationAliases;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopes;
  mutable llvm::SmallVector<llvm::StringMap<std::string>, 8>
      scopedCanonicalBindings;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopedClasses;
};

} // namespace lython::emitter
