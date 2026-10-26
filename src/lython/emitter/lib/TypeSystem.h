#pragma once

#include <string_view>
#include "Ast.h"
#include "Diagnostics.h"
#include "PyDialectTypes.h"
#include "TypeInference.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <memory>

#include <functional>
#include <optional>
#include <string>

namespace lython::emitter {

// The io function open() with a literal binary MODE stands for: the
// Buffered* class CPython's open() returns for it -- BufferedRandom for '+',
// BufferedReader for 'r', BufferedWriter for 'w', 'a' and 'x'
// (runtime/lib/io.py, _open_buffered_*).
inline const char *binaryOpenFunction(std::string_view mode) {
  if (mode.find('+') != std::string_view::npos)
    return "io._open_buffered_random";
  if (mode.find('r') != std::string_view::npos)
    return "io._open_buffered_reader";
  return "io._open_buffered_writer";
}

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
  // `*args: P.args` / `**kwargs: P.kwargs`: the ParamSpec each spells.
  std::optional<std::string> paramSpecArgs;
  std::optional<std::string> paramSpecKwargs;
  // A specialization at one call's shape (ensureGenericSpecialization): what
  // the call gave P are parameters of their own -- unnamed positional ones
  // from `paramSpecPositionalStart`, keyword-only ones from
  // `paramSpecKwOnlyStart` -- with no `*args`/`**kwargs` left in the ABI, and
  // the body binds those names to packs of them, at the declared pack types.
  bool paramSpecExpanded = false;
  unsigned paramSpecPositionalStart = 0;
  unsigned paramSpecPositionalCount = 0;
  unsigned paramSpecKwOnlyStart = 0;
  unsigned paramSpecKwOnlyCount = 0;
  mlir::Type paramSpecArgsType;
  mlir::Type paramSpecKwargsType;
  unsigned positionalOnlyCount = 0;
  mlir::Type resultType;
  mlir::Type publicResultType;
  bool isAsyncFunction = false;
  bool isGeneratorFunction = false;
  bool isAsyncGeneratorFunction = false;
  // An `async def` without `yield`: its body is a generator's -- one that
  // yields `object` (what it awaits hands to the task) and returns the
  // declared result -- and calling it makes a `types.CoroutineType`.
  bool isCoroutineFunction = false;
  // The mismatch MESSAGE, not a flag: the two types are known only here, and
  // "incompatible with inferred Generator contract" left the reader to guess
  // which of the yield, send and return channels disagreed.
  std::string generatorAnnotationMismatch;
  bool asyncGeneratorReturnsValue = false;
  llvm::SmallVector<std::string, 4> generatorAnalysisFailures;
  mlir::Type inferredGeneratorType;
  mlir::Type generatorYieldType;
  // True when `generatorYieldType` is the ANNOTATION's rather than the walk's.
  // Only then is each yield checked against it at its site: for an unannotated
  // generator the walk's answer IS the contract, and coercing to it would wrap
  // a narrowed value back into the union the walk inferred.
  bool generatorYieldTypeIsAnnotated = false;
  mlir::Type generatorSendType;
  mlir::Type generatorReturnType;
};

// A function whose `*args`/`**kwargs` spell a ParamSpec's `P.args`/`P.kwargs`:
// what they hold is whatever each call passes on to the callable P describes.
inline bool forwardsParamSpec(const FunctionSignature &sig) {
  return sig.paramSpecArgs.has_value() || sig.paramSpecKwargs.has_value();
}

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
// An expression that builds an EMPTY container and so carries no element type
// of its own (`[]`, `{}`, `()`, and the zero-argument constructors).
bool isEmptyContainerExpression(const parser::Node *node);

struct ExprInferenceContext {
  const llvm::StringMap<mlir::Type> &localCallables;
  llvm::SmallVectorImpl<std::string> *failureReasons = nullptr;
  // Locals bound so far by a body walk (assignments, loop targets); shadows
  // the symbol table without mutating it.
  const llvm::StringMap<mlir::Type> *localSymbols = nullptr;
  // Non-strict contexts keep the object() fallbacks but still see
  // localCallables/localSymbols.
  bool strict = true;
  // Whether this walk answers about the code the emitter is emitting, and so
  // sees the proofs it holds about fields and cells (`setMemberProofs`). A
  // context derived inside such a walk carries it on; a body walk of some
  // OTHER function does not, since `self.s` there names another object.
  bool seesEmitterProofs = false;
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

// The names one statement spells and the names it binds, anywhere inside it:
// what the empty-literal seed scan reads to find the last statement that can
// matter to it (EmptyLiteralSeed.cpp). `node` is the statement it was read
// from, compared on every use.
struct StatementNames {
  const parser::Node *node = nullptr;
  llvm::StringSet<> mentioned;
  llvm::StringSet<> stored;
};
// Fills `names` from `node`; a def or class is its name and its header.
void collectStatementNames(const parser::Node &node, StatementNames &names);

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
  // What one binding pass wrote: the innermost scope's names and the global
  // annotation aliases it set or dropped, in order. ModuleEmitter keeps one
  // per imported module so that binding a module's imports again -- once per
  // method body inlined from it -- is a copy rather than a second reading.
  struct AliasWrite {
    bool typeAlias = false; // annotationTypeAliases, else annotationAliases
    std::string name;
    mlir::Type type;        // null with typeAlias: the alias was dropped
    std::string target;
  };
  // The `@overload`s each binding of a name declares, keyed by the
  // implementation's type.
  using FunctionOverloadSets = llvm::SmallVector<
      std::pair<mlir::Type, llvm::SmallVector<py::CallableType, 2>>, 1>;
  struct FrameRecord {
    llvm::StringMap<mlir::Type> symbols;
    llvm::StringMap<FunctionOverloadSets> overloads;
    llvm::StringMap<std::string> canonical;
    llvm::StringMap<mlir::Type> classes;
    llvm::StringMap<mlir::Type> typeParameters;
    llvm::StringSet<> moduleNames;
    // The last write to each alias, keyed by table and name: what replaying
    // the whole sequence would leave.
    llvm::StringMap<AliasWrite> aliasWrites;
    // The alias epoch these writes were last applied at (replayFrame).
    mutable std::uint64_t appliedAt = ~std::uint64_t{0};
  };
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
    llvm::SmallVector<llvm::StringSet<>, 8> savedModuleNames;
    llvm::SmallVector<llvm::StringMap<FunctionOverloadSets>, 8>
        savedOverloads;
    std::shared_ptr<const FrameRecord> savedUnderlay;
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
  // The callable a manifest class's method has when it is read off the CLASS
  // and used as a value (`str.lower`). Empty unless the forwarder the emitter
  // synthesizes for it can be written -- see tryEmitManifestMethodObject, which
  // asks this same question so the two answers cannot drift apart.
  std::optional<py::CallableType>
  unboundManifestMethodCallable(mlir::Type typeObject,
                                llvm::StringRef methodName) const;
  // The parameter types a call with `argumentCount` positional arguments
  // meets on a builtin instance's method: the shortest overload that takes
  // that many, past the receiver. What a lambda argument is typed by.
  std::optional<py::CallableType>
  manifestMethodCallExpectations(mlir::Type receiver,
                                 llvm::StringRef methodName,
                                 unsigned argumentCount) const;
  // The callable a method read off an INSTANCE of a builtin class is, as the
  // emitter's forwarder takes it (tryEmitBoundManifestMethodObject): the
  // shortest overload, its required parameters, the receiver bound.
  std::optional<py::CallableType>
  boundManifestMethodCallable(mlir::Type receiver,
                              llvm::StringRef methodName) const;
  // The instance contract behind a type object whose manifest class declares
  // `methodName` as an INSTANCE method -- the receiver an unbound call has to
  // shift its first argument into. Null for a source class, for a staticmethod
  // or classmethod (whose first parameter is not the class), and for a name the
  // manifest does not resolve to exactly one method.
  mlir::Type manifestMethodReceiverContract(mlir::Type typeObject,
                                            llvm::StringRef methodName) const;
  // The emitter's proofs about field paths (`self.s`) and re-read names (a
  // cell, a module global), consulted by the walks that answer about the
  // code being emitted.
  // `suppressed` is the emitter's own switch for a RAW read -- the load a
  // checked read tests before it spends the proof -- which must be typed as
  // the storage is, not as the proof says.
  void setMemberProofs(const llvm::StringMap<mlir::Type> *proofs,
                       const bool *suppressed) {
    memberProofs = proofs;
    memberProofsSuppressed = suppressed;
  }
  // Why a class does not satisfy a protocol, keyed "Class\0Protocol" by
  // contract name (desugarProtocols fills it): what a refused call says when
  // the parameter it missed is a protocol.
  void setProtocolMisses(const llvm::StringMap<std::string> *misses) {
    protocolMisses = misses;
  }
  // True when the manifest class of the instance type `instance` declares
  // `methodName` as a CLASSMETHOD in every overload -- its first parameter is
  // the class object -- so `instance.methodName(...)` is the class's call.
  bool isManifestClassMethod(mlir::Type instance,
                             llvm::StringRef methodName) const;
  // Whether `dict.get(k, default)` over values of `valueType` takes the
  // default AS a value: it has that type, or it is an empty container of the
  // same kind (`{}` for a dict of dicts), which CPython does not tell apart.
  bool dictGetDefaultIsValue(mlir::Type valueType,
                             const parser::Node *defaultNode) const;
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
  void setJsHost(bool value) { jsHost = value; }

  // Module pre-pass: resolves the signatures of all top-level functions in
  // callee-first (SCC topological) order, memoizes them, and binds their
  // public callables so forward references and mutual recursion type-check
  // regardless of declaration order. Emit stays declaration-ordered; it hits
  // the memo through functionSignature.
  void registerModule(const parser::Node &moduleNode);
  // A function defined at the top of a registered module, by name.
  const parser::Node *topLevelFunction(llvm::StringRef name) const {
    auto found = topLevelFunctionDefs.find(name);
    return found == topLevelFunctionDefs.end() ? nullptr : found->second;
  }
  // Whether a direct call of `function`, declared `sig`, is re-read at its
  // arguments' numeric rungs: a plain positional signature with no type
  // parameter and no decorator. The emitter registers exactly these.
  bool argumentSpecializable(const parser::Node &function,
                             const FunctionSignature &sig) const;
  // The signature a call of `function` is read at when `supplied` -- one type
  // per declared positional parameter -- stands a rung below a parameter, or
  // nothing when the call is the declared one. The emitter specializes by it
  // and inference types the call by it, so the two cannot disagree.
  std::optional<FunctionSignature>
  argumentRungSpecialization(const parser::Node &function,
                             const FunctionSignature &declared,
                             llvm::ArrayRef<mlir::Type> supplied) const;

  Scope pushScope() const;
  ScopeIsolation isolateScopes() const;
  // Only one scope deep, and empty: the frame a binding pass starts from.
  bool atFreshIsolatedFrame() const;
  // Logs alias writes into `record` too, until the matching stop. Nested
  // recordings each see the writes of everything inside them.
  void startRecordingAliasWrites(FrameRecord &record) {
    aliasRecords.push_back(&record);
  }
  void stopRecordingAliasWrites() { aliasRecords.pop_back(); }
  void captureFrame(FrameRecord &record) const;
  // ⭐ UNDER the innermost frame, not copied into it: the frame reads what it
  // does not bind itself from the record, as an outer frame would be read.
  // Copying was slower than reading again -- the frames of a module that
  // imports argparse hold thousands of names, replayed ten thousand times.
  void replayFrame(std::shared_ptr<const FrameRecord> record);
  // Why a replay and a fresh reading differ, or empty when they agree.
  std::string compareFrame(const FrameRecord &replayed) const;
  // The annotation alias tables, for the memo's verify mode to compare what
  // a replay leaves with what a reading leaves.
  struct AliasTables {
    llvm::StringMap<std::string> aliases;
    llvm::StringMap<mlir::Type> typeAliases;
  };
  AliasTables aliasTables() const {
    return {annotationAliases, annotationTypeAliases};
  }
  void restoreAliasTables(const AliasTables &tables) {
    annotationAliases = tables.aliases;
    annotationTypeAliases = tables.typeAliases;
    ++aliasEpoch;
  }
  void bindLocalSymbol(llvm::StringRef name, mlir::Type type) const;
  // Monomorphization: the ground type a specialization solved for a type
  // parameter, visible to ANNOTATIONS in the specialized body (`out:
  // list[T] = []`) for the current scope. A plain symbol binding does not
  // serve: annotationTypeForName deliberately ignores value symbols so a
  // local cannot shadow a class annotation.
  // Whether a scope is open that `bindLocalTypeParameter` binds into.
  // The element of `left + right` for two tuples, at least one of unknown
  // length: the join of every element either side holds.
  std::optional<mlir::Type> tupleConcatElement(mlir::Type left,
                                               mlir::Type right) const;
  // A type parameter or a type alias bound under this exact spelling.
  std::optional<mlir::Type> scopedTypeAlias(llvm::StringRef name) const;
  bool inTypeParameterScope() const { return !scopedTypeParameters.empty(); }
  void bindLocalTypeParameter(llvm::StringRef name, mlir::Type type) const;
  void bindSymbol(llvm::StringRef name, mlir::Type type);
  // The MODULE's own symbol table, whatever scope is open. A synthesized
  // module function is visible from every body, and binding it in whichever
  // scope happened to demand it made the name unresolvable from the next one
  // ("unresolved name '__lyvdisp$2'" from a method object taken inside a loop).
  void bindRootSymbol(llvm::StringRef name, mlir::Type type);
  // `bindCanonicalSymbol` outside every scope: a name the module's own scope
  // reads wherever it is asked from (a reserved one; no source name is).
  void bindRootCanonicalSymbol(llvm::StringRef name, llvm::StringRef canonical,
                               mlir::Type type);
  void bindCanonicalSymbol(llvm::StringRef name, llvm::StringRef canonical,
                           mlir::Type type);
  std::optional<mlir::Type> lookupSymbol(llvm::StringRef name) const;
  // Does `name` here spell a class -- not a local or parameter that happens
  // to share a class's spelling?
  bool namesClassInScope(llvm::StringRef name) const;
  std::optional<std::string> lookupCanonicalBinding(llvm::StringRef name) const;
  // `sys.stdout` / `sys.stderr` when the program assigns them
  // (ModuleEmitter::collectReplacedStreams): the binding is then a cell that
  // holds either text stream the runtime has, and its type is their union.
  // Set before any import is bound.
  void setReplacedStreams(llvm::ArrayRef<std::string> canonicalNames);
  bool isReplacedStream(llvm::StringRef canonicalName) const;
  mlir::Type replacedStreamType() const;
  // The callable a program IMPORTED under `name` (`from helpers import
  // range`), which outranks every builtin meaning of the spelling -- its class
  // and the inference's per-name rules alike. Not the builtins' own canonical
  // bindings (`open` is `_io.open`).
  std::optional<py::CallableType> importedCallable(llvm::StringRef name) const;
  void bindClass(llvm::StringRef name, mlir::Type instanceType);
  std::optional<mlir::Type> lookupClass(llvm::StringRef name) const;
  // The hierarchy AS WRITTEN, from the module pre-pass. Registered before any
  // statement is emitted so a question asked above a class answers the same as
  // one asked below it -- the class ops the subtype walk reads are created as
  // each ClassDef is reached, which is why `isinstance(a, B)` in a function
  // defined above `class B(A)` folded to False and ran the wrong branch.
  void bindDeclaredBases(llvm::StringRef name,
                         llvm::ArrayRef<std::string> bases);
  bool declaredSubclassOf(llvm::StringRef sub, llvm::StringRef super) const;
  // Whether this compiler DECLARED the class at all -- written here or
  // imported. The name-shape test ("has no dot") calls an imported class
  // manifest, which is a wrong answer for every hierarchy question about one.
  bool isDeclaredClass(llvm::StringRef name) const {
    return declaredBases.count(name) != 0;
  }
  // The nearest SOURCE class every member derives from, or null when there is
  // none -- what a join of source classes returns instead of a union of them.
  mlir::Type nearestCommonSourceBase(mlir::ArrayRef<mlir::Type> members) const;
  // The nearest class every member derives from through the program's
  // declared bases and the builtin exception taxonomy; null at `object`.
  mlir::Type nearestCommonClass(mlir::ArrayRef<mlir::Type> members) const;
  // `list`, `dict`, `tuple`, `set` and `type` read as values: their class
  // values. Bound nowhere else, because a CALL of each is its own path.
  std::optional<mlir::Type> builtinClassValue(llvm::StringRef name) const;
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
  // ⭐ Property getters (`@property`) resolve `instance.name` to the getter's
  // RETURN type. They are neither a field nor a manifest method, so the
  // attribute inference had nothing to answer with and returned
  // `builtins.object` — which routed `str(p.parent)` into the erased-object
  // dispatch and SEGFAULTED. The emitter registers them per class, the same
  // channel static attributes use.
  void bindClassPropertyType(llvm::StringRef className,
                             llvm::StringRef propertyName, mlir::Type type);
  std::optional<mlir::Type>
  lookupClassPropertyType(llvm::StringRef className,
                          llvm::StringRef propertyName) const;
  // Static methods (`@staticmethod`) take no receiver, so the method-contract
  // channel — which binds parameter 0 to the receiver — cannot resolve them.
  // The emitter registers their signatures here as it emits the class.
  void bindClassStaticMethod(llvm::StringRef className,
                             llvm::StringRef methodName, mlir::Type callable);
  // A module function declared with `@overload`s: a call of `name` bound to
  // `implementation` is typed by the first overload that accepts it.
  void bindFunctionOverloads(llvm::StringRef name, mlir::Type implementation,
                             llvm::SmallVector<py::CallableType, 2> overloads);
  const llvm::SmallVector<py::CallableType, 2> *
  functionOverloads(llvm::StringRef name, mlir::Type implementation) const;
  // The innermost frame no longer knows which overloads `name` has: a path
  // that binds it something else may reach the next read.
  void forgetFunctionOverloads(llvm::StringRef name);
  // An alias a statement of the module body makes. A body emitted as a
  // function of its own runs whenever it is called, when the module name may
  // hold another function, so it does not see these overloads.
  void bindModuleBodyAlias(llvm::StringRef name, mlir::Type implementation,
                           llvm::SmallVector<py::CallableType, 2> overloads);
  struct HiddenOverloads {
    // (frame, or -1 for the root; name; its sets)
    llvm::SmallVector<std::tuple<int, std::string,
                                 llvm::SmallVector<std::pair<
                                     mlir::Type,
                                     llvm::SmallVector<py::CallableType, 2>>,
                                                   1>>,
                      2>
        entries;
  };
  HiddenOverloads hideModuleBodyAliases();
  void restoreModuleBodyAliases(HiddenOverloads hidden);
  // A module name bound once to the module function `target`: it answers
  // `target`'s overloads before the binding is emitted.
  void bindFunctionOverloadAlias(llvm::StringRef name,
                                 llvm::StringRef target);
  std::optional<mlir::Type>
  lookupClassStaticMethod(llvm::StringRef className,
                          llvm::StringRef methodName) const;
  bool bindImportedModule(llvm::StringRef module, llvm::StringRef localName);
  // Whether a name in scope is a MODULE this compiler bound, as opposed to an
  // ordinary value that happens to be `object`-typed. A module binds its local
  // name to `object`, so the two are otherwise indistinguishable -- and the
  // difference decides whether an unresolvable attribute on it is a static
  // error or a dynamic read.
  bool isImportedModuleName(llvm::StringRef name) const;
  void noteImportedModuleName(llvm::StringRef name);
  bool bindImportedName(llvm::StringRef module, llvm::StringRef exportedName,
                        llvm::StringRef localName);

  // ⭐ Drops a memoized signature so the next ask recomputes it. `registerModule`
  // runs before any class contract exists -- it has to, because a signature may
  // name a class and the class's own bodies are typed against the function
  // symbols. A GENERATOR's signature is the one that also depends on its BODY,
  // and a body that reads `c.n` on a source class inferred `object` there.
  void forgetSignature(const parser::Node *function);
  void forgetInferredResult(const parser::Node *function);
  // False only during the module fixpoint's rounds; a signature read there is
  // not the one read after it (see `defaultsDescribeParameters`).
  bool describesParametersByDefaults() const {
    return defaultsDescribeParameters;
  }
  mlir::Type annotationType(const parser::Node *node) const;
  // True when the expression NAMES A TYPE rather than a value: a builtin
  // spelling, a declared class, a protocol or contract name, an alias already
  // bound, or a subscript or `|` union of those. This is what tells
  // `Name = str` (a type alias) from `Name = "str"` (a string).
  bool namesAType(const parser::Node *node) const;
  // Bind a module-level type alias. Unlike `bindAnnotationAlias`, which maps a
  // name to another SPELLING (what an import rename needs), this carries the
  // resolved type, so `Row = list[int]` works as well as `Name = str`.
  void bindAnnotationTypeAlias(llvm::StringRef name, mlir::Type type);
  // The members of a tuple whose arity is known: a display's elements, or a
  // tuple type of two or more positions (one position is the homogeneous
  // `tuple[T]`, which says nothing of its length).
  // `inferred`, when given, is the node's type already inferred by the caller.
  std::optional<llvm::SmallVector<mlir::Type, 8>>
  fixedTupleMembers(const parser::Node *node,
                    mlir::Type inferred = mlir::Type()) const;
  // The `sys.exc_info()` calls written in an `except` block's body: typed as
  // the triple without None, where every other one is typeshed's optional
  // triple (ModuleEmitter::markExcInfoCallsInsideHandlers).
  void markExcInfoInsideHandler(const parser::Node *call) {
    excInfoCallsInsideHandlers.insert(call);
  }
  bool excInfoInsideHandler(const parser::Node *call) const {
    return excInfoCallsInsideHandlers.contains(call);
  }
  mlir::Type optionalExcInfoType() const;
  // Takes back an alias bound for one module's contracts only (a stub's
  // aliases name types inside the stub and are not the program's to spell).
  void unbindAnnotationTypeAlias(llvm::StringRef name);
  // Diagnostics recorded while resolving annotations (string forward
  // references whose text is not a simple name). Drained once by the
  // emitter when it assembles its result.
  parser::Diagnostics takeAnnotationDiagnostics();
  // Puts back what a take set aside, ahead of anything recorded since.
  void restoreAnnotationDiagnostics(parser::Diagnostics diagnostics);
  // The element an iteration over `node` yields (a generator expression or a
  // plain iterable); null when it cannot be seen.
  mlir::Type iterationElementType(const parser::Node *node) const;
  // What iterating a value of `iterableType` yields: `__iter__` then
  // `__next__`, or the sequence protocol. Null when it is not iterable.
  mlir::Type iterableElementType(mlir::Type iterableType) const;
  mlir::Type inferExpr(const parser::Node *node) const;
  mlir::Type inferExpr(const parser::Node *node,
                       const ExprInferenceContext &ctx) const;
  CallInferenceResult
  inferCallWithEvidence(mlir::Type calleeType,
                        mlir::ArrayRef<mlir::Type> positional,
                        mlir::ArrayRef<CallKeywordType> keywords) const;
  // `factory(args)(applied)` where `factory` is generic in a parameter only
  // its RESULT names -- `def skip[F](reason) -> Callable[[F], F]`, the shape
  // of a decorator factory: the two calls solved as one. Nothing when the
  // factory call alone answers, or the shape is not two positional calls.
  struct CurriedCallSolution {
    mlir::Type result;
    // The factory's signature at the solution, and what it returns.
    py::CallableType factory;
    py::CallableType applied;
  };
  std::optional<CurriedCallSolution>
  solveCurriedGenericCall(mlir::Type factoryType,
                          mlir::ArrayRef<mlir::Type> factoryArguments,
                          mlir::ArrayRef<mlir::Type> appliedArguments) const;
  CallInferenceResult inferMethodCallWithEvidence(
      mlir::Type receiverType, llvm::StringRef methodName,
      mlir::ArrayRef<mlir::Type> positional,
      mlir::ArrayRef<CallKeywordType> keywords = {}) const;
  // Manifest fact (`ly.typing.structural_mutators`): the method structurally
  // mutates the receiver, so its call rebinds the receiver local through an
  // extra receiver-typed call result.
  // Does the receiver declare a manifest method of this NAME at all, whatever
  // its signature? "does not provide" and "does not accept these arguments"
  // are different sentences and the reader acts on them differently.
  bool declaresManifestMethod(mlir::Type receiverType,
                              llvm::StringRef methodName) const;
  bool isStructuralMutatorMethod(mlir::Type receiverType,
                                 llvm::StringRef methodName) const;
  // Ordered `__match_args__` attribute names for positional class patterns on
  // the receiver's class; nullopt when the class declares none.
  std::optional<std::vector<std::string>>
  classMatchArgs(mlir::Type receiverType) const;
  AwaitInferenceResult inferAwaitWithEvidence(mlir::Type awaitableType) const;
  YieldFromInferenceResult
  inferYieldFromWithEvidence(mlir::Type sourceType) const;
  mlir::Type inferCall(mlir::Type calleeType,
                       mlir::ArrayRef<mlir::Type> positional,
                       mlir::ArrayRef<CallKeywordType> keywords) const;
  mlir::Type
  inferClassInstantiation(mlir::Type instanceType,
                          mlir::ArrayRef<mlir::Type> positional,
                          mlir::ArrayRef<CallKeywordType> keywords) const;
  mlir::Type join(mlir::ArrayRef<mlir::Type> types) const;
  // The result an OPERATOR has on a union receiver: the join of what each
  // member answers, or null unless every member answers. The emitter builds
  // the matching tag dispatch (`emitUnionOperator`); nothing else may use this
  // without building one, or the two channels disagree.
  mlir::Type unionOperatorResult(mlir::Type receiver, llvm::StringRef method,
                                 mlir::ArrayRef<mlir::Type> arguments) const;
  // The mirror image: a concrete receiver and a UNION argument.
  mlir::Type unionArgumentOperatorResult(mlir::Type receiver,
                                         llvm::StringRef method,
                                         mlir::Type argument) const;
  // The result of `u.method(...)` on a union receiver: the join of what each
  // member's method returns, or null unless every member has one that takes
  // these arguments. Read only where the emitter builds the matching tag
  // dispatch (`tryEmitUnionMethodCall`) -- the inferred type of a method call
  // and that call's emission.
  mlir::Type unionMethodResult(mlir::Type receiver, llvm::StringRef method,
                               mlir::ArrayRef<mlir::Type> positional,
                               mlir::ArrayRef<CallKeywordType> keywords) const;
  // Same-shape callables collapse to one callable whose result is the join of
  // theirs, instead of a union nothing can call.
  mlir::Type commonCallableJoin(mlir::ArrayRef<mlir::Type> members) const;
  mlir::Type widenLiteral(mlir::Type type) const;
  // A union of bool and int (literals widened), with a bool among them: the
  // value a call passes where int is declared as the int it is.
  bool isUnionOfBoolAndInt(mlir::Type type) const;

  // Synthesized-function support (lazy iterator desugars): pins an
  // unannotated parameter node to a concrete type, exactly like the module
  // pre-pass does for inferred parameters — functionSignature reads it when
  // the arg node carries no annotation.
  void overrideParameterType(const parser::Node *argNode, mlir::Type type) {
    parameterTypeOverrides[argNode] = type;
  }
  // And its result, for a synthesized dispatcher whose result is a type no
  // annotation spells (a class of another module, a union): read where an
  // annotation would be.
  void pinResultType(const parser::Node *function, mlir::Type type) {
    pinnedResultTypes[function] = type;
  }

  // `selfType` is the receiver's CONCRETE type, and it matters for more than
  // the signature: the body is inferred inside this call, so with the
  // placeholder py.self bound to the receiver, `self.n` had no attribute to
  // resolve and every unannotated method inferred builtins.object. The class
  // emitter substitutes py.self in the signature afterwards either way --
  // that substitution reaches the types, never the body inference.
  //
  // `monomorphize` re-reads the function AS IF its annotations were absent:
  // the parameters take `expectedCallable`'s types and the return annotation
  // is ignored so the body walk decides the result. That is what an argument
  // specialization needs and what a lambda has always had -- a lambda's
  // parameters already come from the expected callable, because it has no
  // annotations to come from. It is deliberately not the default: an
  // annotation is a CONTRACT at the declaration and only a call that has
  // already been shown to be admissible may re-read it as inert.
  FunctionSignature
  functionSignature(const parser::Node &function,
                    std::optional<llvm::StringRef> selfName = std::nullopt,
                    py::CallableType expectedCallable = {},
                    mlir::Type selfType = {}, bool monomorphize = false) const;
  void refreshCallable(FunctionSignature &sig) const;

  // Each statement of `suite`, read once per suite.
  const std::vector<StatementNames> &
  statementNamesOf(const std::vector<parser::NodePtr> &suite) const;

  // The emitter's reads of an imported module's static constants
  // (`sourceModuleLiteralConstant`), keyed by the module body -- its first
  // statement and its length -- and the name. ⛔ Not recomputed per read:
  // each is a walk of the whole body, and every inlined method of a module's
  // class re-binds that module's names, so `from ._constants import *` with
  // 130 names made the unittest emitter ten times slower.
  llvm::DenseMap<std::pair<const void *, unsigned>,
                 llvm::StringMap<std::optional<mlir::Type>>>
      moduleConstantMemo;
  // A module body's statically taken statements (EmitterImports.cpp), by the
  // body; boxed, so a reference survives the map growing.
  llvm::DenseMap<const void *,
                 std::unique_ptr<std::vector<std::shared_ptr<parser::Node>>>>
      moduleStatementsMemo;

private:
  mlir::Type inferExprImpl(const parser::Node *node,
                           const ExprInferenceContext *ctx) const;
  void popScope() const;
  void bindAnnotationAlias(llvm::StringRef name, llvm::StringRef target);
  std::string resolveAnnotationName(llvm::StringRef name) const;
  mlir::Type annotationTypeForName(llvm::StringRef name) const;

  mlir::MLIRContext &context;
  mutable InferenceContext inferenceState;
  const llvm::StringMap<mlir::Type> *memberProofs = nullptr;
  const llvm::StringMap<std::string> *protocolMisses = nullptr;
  const bool *memberProofsSuppressed = nullptr;
  llvm::StringSet<llvm::MallocAllocator> importedModuleLocalNames;
  // The module names each pushed scope bound, beside `scopes`: a module name
  // is a fact about one binding, and goes when its scope does.
  // ⛔ Not the one set above for every scope: `import posixpath as path`
  // inside os, read in os's own scope, made `path` a module in every module
  // after it, and `def path(): ...; path()` in the program was refused as
  // "module 'path' is not a value".
  mutable llvm::SmallVector<llvm::StringSet<>, 8> scopedModuleNames;
  llvm::SmallVector<FrameRecord *, 2> aliasRecords;
  // A replayed frame below scope 0 (replayFrame); dropped with scope 0.
  mutable std::shared_ptr<const FrameRecord> underlay;
  // Counts the writes to the annotation alias tables.
  std::uint64_t aliasEpoch = 0;
  void logAliasWrite(const AliasWrite &write) {
    for (FrameRecord *record : aliasRecords)
      record->aliasWrites[(llvm::Twine(write.typeAlias ? 'T' : 'A') + "." +
                           write.name)
                              .str()] = write;
  }
  void noteModuleNameInScope(llvm::StringRef name);
  void forgetModuleNameInScope(llvm::StringRef name) const;
  // Signatures resolved by registerModule's pre-pass, keyed by function
  // node. Only module top-level functions are memoized: nested defs,
  // lambdas, and imported source modules run under caller-specific scope
  // contexts a node-keyed cache would conflate.
  llvm::DenseMap<const parser::Node *, FunctionSignature> signatureMemo;
  llvm::StringMap<const parser::Node *> topLevelFunctionDefs;
  // Functions whose numeric RETURN annotation is being re-read against their
  // body right now. A cycle -- direct or mutual recursion -- takes the
  // annotation and stops, which is what the annotation is for.
  mutable llvm::DenseSet<const parser::Node *> returnRungWalks;
  mutable llvm::DenseMap<const std::vector<parser::NodePtr> *,
                         std::vector<StatementNames>>
      suiteStatementNames;
  // ⭐ A DECORATOR IS A CALL THE SOURCE DOES NOT SPELL. `@d def f` is
  // `f = d(f)`, and the parameter fixpoint below reads its constraints off
  // CALL NODES -- so `def d(fn)` with no annotation was refused under the
  // decorator syntax while the same program written as an assignment beside it
  // inferred `fn` fine. The applications are synthesized here and kept alive
  // for as long as the nodes are pointed at.
  std::vector<parser::NodePtr> decoratorCallNodes;
  // Inference variables standing in for missing annotations, assigned by
  // registerModule's pre-pass and resolved by its module-wide fixpoint.
  // Keyed by arg node (parameters) / function node (result). functionSignature
  // reads them through zonk, so provisional and final signature computation
  // share one code path.
  llvm::DenseMap<const parser::Node *, mlir::Type> parameterTypeOverrides;
  // False only while the module fixpoint is still propagating: a parameter
  // whose inference variable is unresolved takes its default's type, but not
  // before the call sites have had their rounds to bind that variable.
  mutable bool defaultsDescribeParameters = true;
  // The call whose type is being asked again with each positional union of
  // bool and int as the int its emission passes, whether this asking is that
  // one, and whether its first asking met such a union (`inferExprImpl`).
  mutable const parser::Node *callAskedAgain = nullptr;
  mutable bool rungUnionsAsInt = false;
  mutable bool callMetRungUnion = false;
  // The functions inference is re-reading at an argument's rung, so a body
  // that calls one of them again takes its declaration instead of re-entering.
  mutable llvm::SmallPtrSet<const parser::Node *, 4> rungSpecializing;
  // What `argumentRungSpecialization` answered, by function and the parameter
  // types it re-read them at; dropped whenever a signature is forgotten.
  mutable llvm::DenseMap<std::pair<const parser::Node *, mlir::Type>,
                         std::optional<FunctionSignature>>
      rungSpecializations;
  // `function` read at `actual`, the parameter types `target` carries, when
  // the emit boundary would take that reading.
  std::optional<FunctionSignature>
  readAtArgumentRungs(const parser::Node &function,
                      const FunctionSignature &declared,
                      llvm::ArrayRef<mlir::Type> actual,
                      py::CallableType target) const;
  // The type of a direct call of the top-level function `name` when it is
  // read at its arguments' rungs, or nothing when it is not.
  std::optional<mlir::Type>
  rungSpecializedCallResult(llvm::StringRef name, const parser::Node &call,
                            llvm::ArrayRef<mlir::Type> positional,
                            llvm::ArrayRef<CallKeywordType> keywords,
                            const ExprInferenceContext *ctx) const;
  llvm::DenseMap<const parser::Node *, mlir::Type> resultTypeOverrides;
  std::string targetTriple;
  bool jsHost = false;
  llvm::StringMap<mlir::Type> symbols;
  llvm::StringSet<> replacedStreams;
  llvm::StringMap<mlir::Type> classes;
  llvm::StringMap<llvm::SmallVector<std::string, 4>> declaredBases;
  // Class static attributes, keyed "<class>.<attr>". The protocol table's
  // field channel models instance layout only, and a static attribute is not
  // a field: it needs its own inference channel so `C.attr` types as the
  // declared attribute rather than falling back to the erased object.
  llvm::StringMap<mlir::Type> classStaticAttrTypes;
  llvm::StringMap<mlir::Type> classPropertyTypes;
  llvm::DenseMap<const parser::Node *, mlir::Type> pinnedResultTypes;
  // Static-method signatures, keyed "<class>.<method>".
  llvm::StringMap<mlir::Type> classStaticMethodTypes;
  // ⛔ Keyed by the name AND the implementation's type, not the type alone:
  // an unrelated function whose signature is the same would be typed by
  // overloads it never declared.
  llvm::StringMap<FunctionOverloadSets> functionOverloadSets;
  llvm::StringMap<std::string> functionOverloadAliases;
  llvm::StringSet<> moduleBodyAliasNames;
  llvm::StringMap<std::string> canonicalBindings;
  // The canonical bindings the builtins scope itself makes, so an import
  // under the same name can be told from them.
  llvm::StringMap<std::string> builtinCanonicalBindings;
  llvm::StringMap<std::string> annotationAliases;
  llvm::StringMap<mlir::Type> annotationTypeAliases;
  llvm::DenseSet<const parser::Node *> excInfoCallsInsideHandlers;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopes;
  mutable llvm::SmallVector<llvm::StringMap<std::string>, 8>
      scopedCanonicalBindings;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8> scopedClasses;
  // ⛔ Per frame, like the canonical bindings, and forgotten when the frame
  // binds the name again: a local `f = other` in a function read the module's
  // `@overload`ed `f` overloads by name and raised on what `other` returned.
  mutable llvm::SmallVector<llvm::StringMap<FunctionOverloadSets>, 8>
      scopedFunctionOverloads;
  mutable llvm::SmallVector<llvm::StringMap<mlir::Type>, 8>
      scopedTypeParameters;
  // Annotation resolution runs from const contexts (registerModule pre-pass
  // and emission may both visit one node), so diagnostics accumulate in a
  // mutable, deduplicated buffer instead of being reported inline.
  mutable parser::Diagnostics annotationDiagnostics;
  // Parsed string annotations, keyed by their text. The TREES are what is
  // cached, not the types: the resolution walks the nodes, so they have to
  // outlive the call, and a second occurrence of the same annotation then
  // costs nothing.
  mutable llvm::StringMap<parser::NodePtr> parsedAnnotations;
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

// The element types a starred argument of `type` contributes, appended to
// `out`; false when the type is not a statically sized tuple. A type-level
// question, so it lives with the type system -- the emitter's support layer
// had a second copy under a second name.
bool appendStarredArgumentTypes(const TypeSystem &types, mlir::Type type,
                                llvm::SmallVectorImpl<mlir::Type> &out);

// Where a type stands in CPython's numeric tower (PEP 484): bool below int
// below float below complex, and -1 for anything that is not one of them (a
// union included).
int numericTowerRung(const TypeSystem &types, mlir::Type type);

// `supplied` when it stands BELOW `declared` on the numeric tower -- a lower
// rung, or a union of rungs none above `declared` and one below it -- and null
// otherwise. A boundary that declares `declared` keeps such a value's own type
// rather than the declaration's: the rungs share no representation, and the
// annotation does not convert in CPython either.
mlir::Type belowDeclaredRung(const TypeSystem &types, mlir::Type supplied,
                             mlir::Type declared);

// The parameter type a call re-reads its callee at when the argument is a
// lower numeric rung than the parameter declares, or null when it is not.
mlir::Type lowerRungParameter(const TypeSystem &types, mlir::Type declared,
                              mlir::Type supplied);

} // namespace lython::emitter
