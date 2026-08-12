#pragma once

#include "Ast.h"
#include "Diagnostics.h"
#include "PyDialectTypes.h"
#include "TypeInference.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
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
  // The mismatch MESSAGE, not a flag: the two types are known only here, and
  // "incompatible with inferred Generator contract" left the reader to guess
  // which of the yield, send and return channels disagreed.
  std::string generatorAnnotationMismatch;
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

class TypeSystem {
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
    friend class TypeSystem;
    explicit Scope(const TypeSystem &owner) : owner(&owner) {}
    void reset();

    const TypeSystem *owner = nullptr;
  };

  // Stashes the ENTIRE pushed-scope stack (root symbols/classes stay
  // visible) and restores it on destruction. Emitting an imported module's
  // body from a use site inside another scope chain re-establishes the
  // defining module's environment with this: a plain pushScope would only
  // shadow, so an unbound name in the imported body could silently resolve
  // to a use-site local instead of being diagnosed.
  class ScopeIsolation {
  public:
    ScopeIsolation() = default;
    ScopeIsolation(const ScopeIsolation &) = delete;
    ScopeIsolation &operator=(const ScopeIsolation &) = delete;
    ScopeIsolation(ScopeIsolation &&other) noexcept;
    ScopeIsolation &operator=(ScopeIsolation &&other) noexcept;
    ~ScopeIsolation();

  private:
    friend class TypeSystem;
    explicit ScopeIsolation(const TypeSystem &owner) : owner(&owner) {}
    void reset();

    const TypeSystem *owner = nullptr;
    llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> savedScopes;
    llvm::SmallVector<llvm::StringMap<std::string>, 8>
        savedCanonicalBindings;
    llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> savedClasses;
    llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> savedTypeParameters;
  };

  explicit TypeSystem(mlir::MLIRContext &context);

  mlir::MLIRContext &getContext() const { return context; }
  void seedBuiltins();

  // Algorithm J unification store. Mutable through const methods for the
  // same reason the scope stacks are: inference state is monotonic engine
  // bookkeeping, not part of the facade's logical constness.
  InferenceContext &inference() const { return inferenceState; }

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

  // Module pre-pass: resolves the signatures of all top-level functions in
  // callee-first (SCC topological) order, memoizes them, and binds their
  // public callables so forward references and mutual recursion type-check
  // regardless of declaration order. Emit stays declaration-ordered; it hits
  // the memo through functionSignature.
  void registerModule(const parser::Node &moduleNode);

  Scope pushScope() const;
  ScopeIsolation isolateScopes() const;
  void bindLocalSymbol(llvm::StringRef name, mlir::Type type) const;
  // Monomorphization: the ground type a specialization solved for a type
  // parameter, visible to ANNOTATIONS in the specialized body (`out:
  // list[T] = []`) for the current scope. A plain symbol binding does not
  // serve: annotationTypeForName deliberately ignores value symbols so a
  // local cannot shadow a class annotation.
  void bindLocalTypeParameter(llvm::StringRef name, mlir::Type type) const;
  void bindSymbol(llvm::StringRef name, mlir::Type type);
  void bindCanonicalSymbol(llvm::StringRef name, llvm::StringRef canonical,
                           mlir::Type type);
  std::optional<mlir::Type> lookupSymbol(llvm::StringRef name) const;
  std::optional<std::string> lookupCanonicalBinding(llvm::StringRef name) const;
  void bindClass(llvm::StringRef name, mlir::Type instanceType);
  std::optional<mlir::Type> lookupClass(llvm::StringRef name) const;
  // Monomorphization of `class C[T]`: the py ABI has no runtime
  // representation for a type parameter, so a generic class is never a
  // contract of its own — every ground instantiation becomes a separate
  // class contract. TypeSystem resolves the SPELLING (`C[int]` in an
  // annotation or in expression position) and hands the base name plus
  // ground arguments to the emitter, which owns the specialization registry
  // and the emission. A null return means "not a generic class, or the
  // arguments are not ground", and the caller keeps its parameterized
  // reading so the ABI's typevar rejection still catches a missed
  // specialization.
  using GenericClassResolver = std::function<mlir::Type(
      llvm::StringRef baseName, mlir::ArrayRef<mlir::Type> arguments)>;
  void setGenericClassResolver(GenericClassResolver resolver);
  // The specialized contract for `baseName[arguments]`, or null.
  mlir::Type resolveGenericClass(llvm::StringRef baseName,
                                 mlir::ArrayRef<mlir::Type> arguments) const;
  // The specialized contract a `C[int]` subscript spells, or null when the
  // node is an ordinary value subscript. Shared by annotation resolution and
  // expression inference (`C[int]` in value position is a class object).
  mlir::Type genericClassSubscript(const parser::Node *node) const;
  // Registers what `C(args)` needs to recover its type arguments without an
  // explicit `C[int]` or an annotated context: the parameter names and the
  // constructor whose parameter types they occur in. Without this a generic
  // class could only be instantiated through a spelled-out instantiation,
  // which no CPython source writes. `fields` stands in for `initNode` when
  // the class has no `__init__` of its own — a dataclass or NamedTuple takes
  // its annotated fields positionally, and that synthesized constructor is
  // the only place its type arguments appear.
  using GenericClassField = std::pair<std::string, const parser::Node *>;
  void registerGenericClass(llvm::StringRef contractName,
                            llvm::ArrayRef<std::string> params,
                            const parser::Node *initNode,
                            llvm::ArrayRef<GenericClassField> fields);
  // Class static attributes (`class C: attr = ...`) type `C.attr` and
  // `instance.attr` reads; the emitter registers them per class as it emits
  // the class contract.
  void bindClassStaticAttr(llvm::StringRef className, llvm::StringRef attrName,
                           mlir::Type type);
  std::optional<mlir::Type>
  lookupClassStaticAttrType(llvm::StringRef className,
                            llvm::StringRef attrName) const;
  // Static methods (`@staticmethod`) take no receiver, so the method-contract
  // channel — which binds parameter 0 to the receiver — cannot resolve them.
  // The emitter registers their signatures here as it emits the class.
  void bindClassStaticMethod(llvm::StringRef className,
                             llvm::StringRef methodName, mlir::Type callable);
  std::optional<mlir::Type>
  lookupClassStaticMethod(llvm::StringRef className,
                          llvm::StringRef methodName) const;
  bool bindImportedModule(llvm::StringRef module, llvm::StringRef localName);
  bool bindImportedName(llvm::StringRef module, llvm::StringRef exportedName,
                        llvm::StringRef localName);

  mlir::Type annotationType(const parser::Node *node) const;
  // Diagnostics recorded while resolving annotations (string forward
  // references whose text is not a simple name). Drained once by the
  // emitter when it assembles its result.
  parser::Diagnostics takeAnnotationDiagnostics();
  // The element an iteration over `node` yields (a generator expression or a
  // plain iterable); null when it cannot be seen.
  mlir::Type iterationElementType(const parser::Node *node) const;
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

  // Synthesized-function support (lazy iterator desugars): pins an
  // unannotated parameter node to a concrete type, exactly like the module
  // pre-pass does for inferred parameters — functionSignature reads it when
  // the arg node carries no annotation.
  void overrideParameterType(const parser::Node *argNode, mlir::Type type) {
    parameterTypeOverrides[argNode] = type;
  }

  // `selfType` is the receiver's CONCRETE type, and it matters for more than
  // the signature: the body is inferred inside this call, so with the
  // placeholder py.self bound to the receiver, `self.n` had no attribute to
  // resolve and every unannotated method inferred builtins.object. The class
  // emitter substitutes py.self in the signature afterwards either way --
  // that substitution reaches the types, never the body inference.
  FunctionSignature
  functionSignature(const parser::Node &function,
                    std::optional<llvm::StringRef> selfName = std::nullopt,
                    py::CallableType expectedCallable = {},
                    mlir::Type selfType = {}) const;
  void refreshCallable(FunctionSignature &sig) const;

private:
  mlir::Type inferExprImpl(const parser::Node *node,
                           const ExprInferenceContext *ctx) const;
  void popScope() const;
  void bindAnnotationAlias(llvm::StringRef name, llvm::StringRef target);
  std::string resolveAnnotationName(llvm::StringRef name) const;
  mlir::Type annotationTypeForName(llvm::StringRef name) const;

  mlir::MLIRContext &context;
  mutable InferenceContext inferenceState;
  // Signatures resolved by registerModule's pre-pass, keyed by function
  // node. Only module top-level functions are memoized: nested defs,
  // lambdas, and imported source modules run under caller-specific scope
  // contexts a node-keyed cache would conflate.
  llvm::DenseMap<const parser::Node *, FunctionSignature> signatureMemo;
  // Inference variables standing in for missing annotations, assigned by
  // registerModule's pre-pass and resolved by its module-wide fixpoint.
  // Keyed by arg node (parameters) / function node (result). functionSignature
  // reads them through zonk, so provisional and final signature computation
  // share one code path.
  llvm::DenseMap<const parser::Node *, mlir::Type> parameterTypeOverrides;
  llvm::DenseMap<const parser::Node *, mlir::Type> resultTypeOverrides;
  std::string targetTriple;
  llvm::StringMap<mlir::Type> symbols;
  llvm::StringMap<mlir::Type> classes;
  // Class static attributes, keyed "<class>.<attr>". The protocol table's
  // field channel models instance layout only, and a static attribute is not
  // a field: it needs its own inference channel so `C.attr` types as the
  // declared attribute rather than falling back to the erased object.
  llvm::StringMap<mlir::Type> classStaticAttrTypes;
  // Static-method signatures, keyed "<class>.<method>".
  llvm::StringMap<mlir::Type> classStaticMethodTypes;
  llvm::StringMap<std::string> canonicalBindings;
  llvm::StringMap<std::string> annotationAliases;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopes;
  mutable llvm::SmallVector<llvm::StringMap<std::string>, 8>
      scopedCanonicalBindings;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopedClasses;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8>
      scopedTypeParameters;
  // Annotation resolution runs from const contexts (registerModule pre-pass
  // and emission may both visit one node), so diagnostics accumulate in a
  // mutable, deduplicated buffer instead of being reported inline.
  mutable parser::Diagnostics annotationDiagnostics;
  GenericClassResolver genericClassResolver;
  struct GenericClassTemplate {
    llvm::SmallVector<std::string, 4> params;
    const parser::Node *initNode = nullptr;
    llvm::SmallVector<GenericClassField, 8> fields;
  };
  llvm::StringMap<GenericClassTemplate> genericClassTemplates;
  // Solves `C(args)`'s type arguments by matching the argument types against
  // `__init__`'s parameter types, with the parameters standing as TypeVars.
  // Null when the constructor does not mention every parameter (`Stack()`
  // determines nothing) — the use site then needs an annotated context.
  mlir::Type
  solveGenericClassInstantiation(llvm::StringRef contractName,
                                 mlir::ArrayRef<mlir::Type> positional,
                                 mlir::ArrayRef<CallKeywordType> keywords) const;
};

} // namespace lython::emitter
