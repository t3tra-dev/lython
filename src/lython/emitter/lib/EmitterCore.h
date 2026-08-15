#pragma once

#include "Emitter.h"
#include "EmitterState.h"
#include "TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace lython::emitter {

class ModuleEmitter {
public:
  ModuleEmitter(const parser::Node &moduleNode, mlir::MLIRContext &context,
                std::string moduleName, std::string sourceName,
                EmitOptions options = {});

  EmitResult emit();

private:
  mlir::Location loc(const parser::Node &node) const;
  mlir::Type callableProtocol() const;
  mlir::Type callProtocolFor(mlir::Type calleeType) const;
  mlir::Type callProtocolFor(const CallInferenceResult &inference,
                             mlir::Type fallback = {}) const;
  bool requireStaticEvidence(const parser::Node &anchor,
                             const CallInferenceResult &inference);
  bool requireStaticEvidence(const parser::Node &anchor,
                             const AwaitInferenceResult &inference);
  bool requireStaticEvidence(const parser::Node &anchor,
                             const YieldFromInferenceResult &inference);
  bool requireStaticEvidence(const parser::Node &anchor,
                             const AsyncIterationInferenceResult &inference);
  bool
  requireStaticEvidence(const parser::Node &anchor,
                        const AsyncContextMethodInferenceResult &inference);
  void predeclareTopLevel();
  void predeclareSourceModules();
  void emitTopLevelDeclarations();
  void emitSourceModuleDeclarations();
  void bindSourceModuleLocals(llvm::StringRef moduleName,
                              const parser::Node &sourceModule, bool isStub);
  void bindModuleImportScope(const parser::Node &sourceModule,
                             bool diagnoseUnsupported);
  bool bindImportStatement(const parser::Node &statement,
                           bool diagnoseUnsupported);
  const EmitOptions::SourceModule *
  lookupSourceModule(llvm::StringRef module) const;
  bool isStubSourceModuleSymbol(llvm::StringRef symbol) const;
  bool bindSourceModuleNamespace(llvm::StringRef module,
                                 llvm::StringRef localName,
                                 unsigned namespaceDepth = 0);
  bool bindSourceModuleName(llvm::StringRef module,
                            llvm::StringRef exportedName,
                            llvm::StringRef localName,
                            unsigned aliasDepth = 0);
  bool bindSourceModuleReexport(const EmitOptions::SourceModule &source,
                                llvm::StringRef exportedName,
                                llvm::StringRef localName);
  bool bindSourceModuleStar(llvm::StringRef module,
                            const parser::Node &anchor,
                            bool diagnoseUnsupported);
  // `from <native manifest module> import *`: expands to the manifest's
  // public exports (callables, classes, constants) — the manifest declares
  // no __all__, so the public-name convention (no leading underscore) is the
  // export list.
  bool bindNativeModuleStar(llvm::StringRef module, const parser::Node &anchor,
                            bool diagnoseUnsupported);
  void bindNativeModuleNamespaceStar(llvm::StringRef module,
                                     llvm::StringRef localName);
  void emitFunctionDecl(const parser::Node &function);
  void emitCallableFunction(const parser::Node &callable,
                            llvm::StringRef symbolName,
                            const FunctionSignature &sig,
                            llvm::ArrayRef<Capture> captures, bool isLambda,
                            unsigned positionalNodeOffset = 0,
                            mlir::Type preboundTypeObject = {});
  std::optional<MethodBinding>
  lookupClassMethod(mlir::Type receiverType, llvm::StringRef methodName) const;
  std::optional<mlir::Type> lookupClassField(mlir::Type receiverType,
                                             llvm::StringRef fieldName) const;
  std::optional<mlir::Type>
  lookupClassStaticAttr(mlir::Type receiverType,
                        llvm::StringRef attrName) const;
  // Canonical contract name for a base-class spelling (resolves import
  // aliases through the class binding; manifest classes resolve to their
  // builtins.* contract). Falls back to the raw spelling.
  std::string canonicalClassName(llvm::StringRef spelling) const;
  // The class's C3 linearization (contract names, self first). Computed and
  // cached by emitClassContract; empty for unknown classes.
  llvm::ArrayRef<std::string> classMro(llvm::StringRef className) const;
  // First class at or after `startAfter` (exclusive when set) in
  // receiverClass's MRO that declares `methodName`; the binding's
  // definingClass names the provider.
  std::optional<MethodBinding>
  resolveMroMethod(llvm::StringRef receiverClass, llvm::StringRef methodName,
                   llvm::StringRef startAfter = {}) const;
  // True when the class (by contract name) linearizes onto a manifest
  // exception class (its instances use the runtime exception representation).
  bool isExceptionBackedClass(llvm::StringRef className) const;
  // True when a subclass of `receiverClass` declares `methodName` itself, so
  // the static class is not enough to say which body a call runs
  // (EmitterClasses.cpp).
  bool subclassOverridesMethod(llvm::StringRef receiverClass,
                               llvm::StringRef methodName) const;
  // The same question about a class-level BINDING rather than a method
  // (EmitterClasses.cpp).
  bool subclassShadowsAttribute(llvm::StringRef receiverClass,
                                llvm::StringRef attributeName) const;
  bool subclassRedeclares(const llvm::StringMap<llvm::StringSet<>> &declarations,
                          llvm::StringRef receiverClass,
                          llvm::StringRef name) const;
  // The one gate every method dispatch goes through: emits the diagnostic and
  // returns true when the receiver's static class cannot answer the call
  // (EmitterClasses.cpp).
  bool refuseUnresolvableDispatch(const parser::Node &anchor, Value receiver,
                                  llvm::StringRef methodName,
                                  const parser::Node *receiverNode = nullptr,
                                  bool throughSuper = false);
  // True when `type` is a SOURCE class whose linearization provides
  // `methodName` only through builtins.object — i.e. it inherits object's
  // default. The question is not "does the method resolve" (since the class's
  // protocol-table entry now has object as a base, it always does) but
  // "does anything before object answer it", which is what decides whether the
  // default may be materialized in place of a dispatch.
  bool inheritsObjectDefaultDunder(mlir::Type type,
                                   llvm::StringRef methodName) const;
  // Of object's eleven declared methods, the six whose default Lython actually
  // provides for an inheriting class. The other five (__init__, __new__,
  // __getattribute__, __setattr__, __delattr__) resolve through the same base
  // and are refused at the call site.
  static bool isImplementedObjectDefault(llvm::StringRef methodName);
  // `x.__str__()` on a class that overrides neither __str__ nor __repr__.
  // CPython's object.__str__ IS type(x).__repr__, so this resolves __repr__ —
  // the source one when there is one, otherwise the address form. Routing it
  // to the manifest object.__str__ instead would print "<object object at
  // ...>", dropping the class name. Nullopt when the default does not apply.
  std::optional<Value> emitInheritedObjectStr(const parser::Node &anchor,
                                              Value receiver);
  // Defining class + storage type of a slot-backed (mutable) class attribute
  // reachable from `className` along its MRO.
  std::optional<std::pair<llvm::StringRef, mlir::Type>>
  resolveClassAttrSlot(llvm::StringRef className,
                       llvm::StringRef attrName) const;
  // Evaluates the class body's attribute initializers into their global
  // slots; runs at the ClassDef statement position in module flow.
  void emitClassAttrInitializers(const parser::Node &classDef);
  Value emitNestedFunctionDecl(const parser::Node &function);
  mlir::ArrayAttr emitCallableDefaultValues(const parser::Node &function,
                                            const FunctionSignature &sig,
                                            llvm::StringRef symbolName);
  Value emitLambda(const parser::Node &expr, py::CallableType expected = {});
  void emitClassContract(const parser::Node &classDef,
                         llvm::StringRef symbolName = {});
  // Monomorphization of `class C[T]`, the class-side counterpart of
  // GenericFunctionInfo: the generic class itself is never emitted (a py
  // class contract has no runtime slot for a type parameter, and its field
  // types and method signatures would carry `!py.typevar` into the ABI).
  // Every ground instantiation the program spells becomes a full, ordinary
  // class contract of its own, so every layer below the emitter — MRO,
  // evidence, refcounting, lowering — keeps seeing only ground classes.
  struct GenericClassInfo {
    const parser::Node *node = nullptr;
    // Type parameter names in declaration order. Kinds other than TypeVar
    // (ParamSpec/TypeVarTuple) are rejected at the instantiation site.
    llvm::SmallVector<std::string, 4> params;
    bool hasPackParameter = false;
    // Specialization contract names derive from this ("C" for main-module
    // classes, "<module>.C" for imported ones), matching the non-generic
    // class symbol scheme so two modules' same-named generics cannot
    // collide.
    std::string symbolBase;
    // Defining source module for imported generics (null for main-module
    // ones): a specialization's method bodies must emit under the DEFINING
    // module's scope, not the use site's.
    const EmitOptions::SourceModule *source = nullptr;
    // Keyed by the parameterized contract `C[args...]` — a uniqued type, so
    // it is the instantiation's identity.
    llvm::DenseMap<mlir::Type, std::string> specializations;
  };
  // The specialized contract for `baseName[arguments]`, or null when
  // baseName is not a registered generic class. Installed as TypeSystem's
  // generic-class resolver, so every annotation and expression that spells
  // an instantiation funnels through here.
  mlir::Type ensureGenericClassSpecialization(
      llvm::StringRef baseName, mlir::ArrayRef<mlir::Type> arguments);
  // Emits specializations allocated but not yet emitted, to a fixpoint (a
  // specialization's own body may spell further ones). `onlyBase` restricts
  // the drain to one generic class: the declaration walk drains at each
  // generic class's own statement position, so a specialization's base-class
  // facts are registered before its MRO is linearized.
  void drainGenericClassSpecializations(llvm::StringRef onlyBase = {});
  void emitGenericClassSpecialization(GenericClassInfo &generic,
                                      llvm::StringRef symbol,
                                      mlir::ArrayRef<mlir::Type> arguments);
  // Registers `class C[T]` for monomorphization instead of emitting it.
  // Returns false when the class is not generic (the caller emits it).
  bool registerGenericClass(const parser::Node &classDef,
                            llvm::StringRef symbolBase,
                            const EmitOptions::SourceModule *source);
  GenericClassInfo *lookupGenericClass(llvm::StringRef name);
  // Binds a specialization's type arguments by their parameter names into the
  // CURRENT scope. Class method bodies are emitted twice over: once as
  // standalone symbols under the specialization's own emission scope, and
  // again inlined at every use site, where that scope is long gone — so the
  // bindings have to be recoverable from the class contract name alone.
  void bindClassTypeArguments(llvm::StringRef className);
  // Diagnoses a bare reference to a generic class where no ground
  // instantiation can be recovered — the class-side counterpart of the
  // generic function's "requires a call or an annotated Callable context".
  void diagnoseUngroundedGenericClass(const parser::Node &anchor,
                                      llvm::StringRef name);
  // Rejects materializing a generic class's class OBJECT (`C.attr`, `C` as a
  // value): monomorphization leaves one contract per instantiation and no
  // class object for the generic itself. Returns nullopt when the class is
  // not generic.
  std::optional<Value> rejectGenericClassObject(const parser::Node &anchor,
                                                mlir::Type classType);
  // The specialized contract a bare `C(...)` should construct, recovered
  // from an expected type that is one of C's specializations; null when the
  // call is not a bare generic-class construction or the expectation does
  // not name an instantiation of it.
  mlir::Type expectedGenericClassInstantiation(const parser::Node &call,
                                               mlir::Type expected);
  // The specialized contract a bare `C(...)` should construct, recovered from
  // the ARGUMENT types through __init__'s parameter types; null when they do
  // not determine every type parameter.
  mlir::Type inferredGenericClassInstantiation(const parser::Node &call);
  void collectClassFields(const parser::Node &classDef,
                          llvm::SmallVectorImpl<std::string> &fieldNames,
                          llvm::SmallVectorImpl<mlir::Type> &fieldTypes,
                          bool includeAnnAssignDefaults = false);
  void collectStaticClassAssignments(
      const parser::Node &classDef, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values,
      llvm::SmallVectorImpl<mlir::Type> *types = nullptr);
  void collectStaticModuleAssignments(
      const parser::Node &moduleNode, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values) const;
  void collectModuleGlobals(const parser::Node &moduleNode);
  // Every name any function in `scope` declares `global`, at any depth.
  llvm::StringSet<> moduleGlobalDeclarations(const parser::Node &scope) const;
  bool isModuleGlobalRead(llvm::StringRef name) const;
  bool isModuleGlobalWrite(llvm::StringRef name) const;
  // Where a structural mutation's re-described receiver goes: a local's
  // binding or the module global's cell (EmitterGlobals.cpp).
  bool isStructuralMutationRebindable(llvm::StringRef name,
                                      mlir::Value receiver) const;
  void rebindStructuralMutation(const parser::Node &at, llvm::StringRef name,
                                Value rebound);
  // Mark a py.global.get/set that reaches the MODULE-GLOBAL population, so
  // lowering gives an `int` one the boxed object cell instead of the machine
  // word. Default cells and class-attribute slots ride the same two ops and
  // are deliberately left unmarked.
  void markBoxedModuleGlobal(mlir::Operation *op) const;

  // EmitterEnums.cpp: `class C(Enum)` desugars to a plain class whose members
  // are class attributes instantiated at the ClassDef statement position, plus
  // synthesized __init__/__str__/__repr__/__eq__ and by-value/by-name lookup
  // classmethods. Runs before any predeclaration so every later layer sees
  // ordinary Python.
  enum class EnumKind { Plain, Int, Str };
  struct EnumMember {
    std::string name;
    bool isStr = false;
    std::int64_t intValue = 0;
    std::string strValue;
    // An alias shares an earlier member's value: it binds to that member's
    // singleton and never appears in iteration (CPython's canonicalization).
    bool isAlias = false;
    std::string aliasOf;
  };
  struct EnumInfo {
    EnumKind kind = EnumKind::Plain;
    std::string name;
    llvm::SmallVector<EnumMember, 8> members;
  };
  void desugarEnumClasses(const parser::Node &moduleNode);
  std::optional<EnumKind> enumBaseKind(const parser::Node &classDef) const;
  void collectEnumMembers(const parser::Node &classDef, EnumKind kind);
  void rewriteEnumClassDef(const parser::Node &classDef);
  void rewriteEnumUses(const parser::Node &node);
  const EnumInfo *enumInfoForNameNode(const parser::Node *node) const;
  parser::NodePtr enumMemberListNode(const EnumInfo &info,
                                     parser::SourceRange range) const;
  llvm::StringMap<EnumInfo> enumClasses;
  // Classes whose `__eq__` is the SYNTHESIZED dataclass one, which answers
  // False for any other class. Only those may have a cross-class comparison
  // folded to a constant (EmitterExpressions.cpp).
  llvm::StringSet<> classesWithClassGuardedEq;

  void emitStatements(const std::vector<parser::NodePtr> *statements,
                      bool skipDeclarations = false);
  void emitStatement(const parser::Node &statement);
  void emitPendingDefaultCells(const parser::Node &statement);
  void emitDelete(const parser::Node &statement);
  void emitAssignTarget(const parser::Node &target, Value value);
  void emitIf(const parser::Node &statement);
  void emitMatch(const parser::Node &statement);
  void emitFor(const parser::Node &statement);
  void emitGeneratorExpFor(const parser::Node &statement,
                           const parser::Node &genexpr);
  // EmitterIterators.cpp: lazy-iterator loop fusion (enumerate/zip/map/
  // filter/reversed/iter and dict view methods in for-iterable position).
  struct LazyCallable {
    parser::NodePtr callee; // Name/Attribute spelling, re-used per element
    llvm::SmallVector<std::string, 3> lambdaParams;
    parser::NodePtr lambdaBody;
  };
  bool tryEmitLazyIteratorFor(const parser::Node &statement,
                              const parser::Node &iterCall);
  bool lazyCallableParts(const parser::Node &statement,
                         const parser::NodePtr &callee, LazyCallable &result);
  bool buildLazyCall(const parser::Node &statement,
                     const LazyCallable &callable,
                     std::vector<parser::NodePtr> arguments,
                     std::vector<parser::NodePtr> &prologue,
                     parser::NodePtr &out);
  bool isBuiltinIteratorName(llvm::StringRef name) const;
  // True when the PROGRAM binds this spelling, so a compiler-known meaning of
  // the same spelling -- a builtin fast path, a builtin class contract -- must
  // not claim the call. Every builtin fast path gates on this one predicate:
  // gating on `values` alone made the winner depend on argument count, because
  // a top-level `def` is absent from `values` and the fast paths are selected
  // by arity.
  bool programBindsName(llvm::StringRef name) const;
  // Collects moduleFunctionNames / moduleClassNames / shadowedBuiltinSymbols.
  void collectTopLevelBindings();
  // The symbol a main-module top-level `def` of this spelling is emitted
  // under: the spelling itself unless it collides with a manifest builtin.
  llvm::StringRef topLevelFunctionSymbol(llvm::StringRef name) const;
  // True when walking `x[0], x[1], ...` up to `len(x)` reproduces iteration
  // over X. The single gate for every index-walk rewrite in EmitterIterators,
  // in both loop and value position.
  bool hasIndexWalkableEvidence(mlir::Type type);
  bool hasIndexableEvidence(const parser::Node *expr);
  // `a, b = xs`: the length comparison CPython's UNPACK_SEQUENCE makes
  // (EmitterStatements.cpp, beside the assignment target walk it guards).
  void emitUnpackArityCheck(const parser::Node &target, Value source,
                            std::size_t expected);
  void runWithScratchNames(llvm::ArrayRef<std::string> names,
                           llvm::function_ref<void()> emit);
  // Value form: enumerate/zip/map/filter/reversed/iter as first-class lazy
  // values synthesize per-call-site generator functions over indexable
  // sequences (memoized by builtin + argument types + callable spelling).
  std::optional<Value>
  tryEmitLazyIteratorValueCall(const parser::Node &expr,
                               const parser::Node *calleeNode);
  struct LazyIteratorSynthesis {
    std::string symbol;
    mlir::Type callableType;
  };
  std::map<std::string, LazyIteratorSynthesis> lazyIteratorMemo;
  std::vector<parser::NodePtr> synthesizedIteratorDefs;
  // itertools desugars (EmitterIterators.cpp). The itertools manifest
  // declares the module contract only; every call compiles here to a loop
  // fusion (for position) or a synthesized generator (value position), so
  // no itertools call may fall through to generic dispatch.
  std::optional<std::string>
  itertoolsCalleeName(const parser::Node *calleeNode);
  bool tryEmitItertoolsFor(const parser::Node &statement,
                           const parser::Node &iterCall);
  std::optional<Value>
  tryEmitItertoolsValueCall(const parser::Node &expr,
                            const parser::Node *calleeNode);
  // dict method sugar (EmitterIterators.cpp): get(k) / setdefault / popitem
  // / dict.fromkeys compose over existing dict primitives.
  bool isDictTypedExpr(const parser::Node *expr);
  bool exprHasContract(const parser::Node *expr, llvm::StringRef contractName);
  std::optional<Value> tryEmitDictMethodSugar(const parser::Node &expr,
                                              const parser::Node *calleeNode);
  // `x in d.keys()/values()/items()` and `len(d.keys())` rewrite against the
  // dict itself.
  std::optional<Value> tryEmitDictViewMembership(const parser::Node &expr);
  // str.maketrans / str.translate compositions (EmitterIterators.cpp).
  std::optional<Value>
  tryEmitStrTranslateSugar(const parser::Node &expr,
                           const parser::Node *calleeNode);
  // sorted(key=, reverse=) / list.sort(key=, reverse=):
  // decorate-sort-undecorate over the native stable sort.
  std::optional<Value> tryEmitSortSugar(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<std::string>
  emitDsuSortStatements(const parser::Node &anchor, parser::NodePtr source,
                        const LazyCallable *key, bool reverse, unsigned serial,
                        llvm::SmallVectorImpl<std::string> &scratchNames);
  void emitWhile(const parser::Node &statement);
  void emitAsyncFor(const parser::Node &statement);
  llvm::SmallVector<CarriedLoopLocal, 4>
  collectCarriedLoopLocals(const parser::Node &statement,
                           const llvm::StringSet<> *excludedNames,
                           llvm::SmallVectorImpl<mlir::Value> &initialValues);
  Value pinLoopCarriedTensor(llvm::StringRef name, Value value,
                             const parser::Node &anchor);
  void bindCarriedLoopLocals(llvm::ArrayRef<CarriedLoopLocal> carried,
                             mlir::Block *block);
  llvm::SmallVector<mlir::Value, 4>
  carriedLoopEdgeOperands(const parser::Node &anchor,
                          llvm::ArrayRef<CarriedLoopLocal> carried,
                          mlir::Block *headerBlock,
                          llvm::ArrayRef<mlir::Value> baselineValues = {},
                          bool toHeader = true);
  llvm::SmallVector<mlir::Value, 4>
  loopCarriedBranchOperands(const parser::Node &anchor,
                            const LoopControlContext &loop, mlir::Block *target);
  void emitTry(const parser::Node &statement);
  // The type a py.try post-try result lane can carry a rebound local out in,
  // or null when the lane cannot carry it and the value has to travel storage
  // instead. Shared by the promotion decision (taken before the regions are
  // emitted) and the lane construction (taken after), which must agree: a name
  // that neither promotes nor lanes reverts to its pre-try value silently.
  mlir::Type postTryLaneCarrierType(mlir::Type type) const;
  void emitTryStar(const parser::Node &statement);
  void emitWith(const parser::Node &statement, bool async);
  void emitWithCleanup(const parser::Node &anchor, const WithCleanup &cleanup);
  // The exception arm's tail: suppress on a truthy __exit__, rethrow
  // otherwise. Null on the normal-path cleanup, which has no decision to take.
  void emitWithExitDecision(const parser::Node &anchor, const Value *suppress);
  void refuseUnrepresentableExitArguments(const parser::Node &anchor,
                                          const MethodBinding &exit);
  void emitWithEnter(const parser::Node &item, bool async);

  mlir::Value emitValueDiamond(mlir::Location location, mlir::Value condition,
                               mlir::Type resultType,
                               llvm::function_ref<mlir::Value()> emitThen,
                               llvm::function_ref<mlir::Value()> emitElse);
  Value emitExpr(const parser::Node *expr);
  // Algorithm M checking mode: emits expr against a downward expected type.
  // Only node kinds whose emitted TYPE depends on the expectation (lambda,
  // container literals) dispatch specially; everything else synthesizes via
  // emitExpr and the caller's coercion/contract check keeps the boundary.
  Value emitExprExpected(const parser::Node *expr, mlir::Type expected);
  Value emitConstant(const parser::Node &expr);
  Value emitCall(const parser::Node &expr);
  // Monomorphization: generic (statically type-parameterized) top-level
  // functions have no direct emission — the py ABI has no runtime
  // representation for a type parameter. Each ground instantiation demanded
  // by a call site or a ground-typed reference emits one specialized copy,
  // deduplicated by instantiated public callable.
  struct GenericFunctionInfo {
    const parser::Node *node = nullptr;
    FunctionSignature signature;
    // Specialization symbols derive from this ("<name>" for main-module
    // functions, "<module>.<name>" for imported ones) instead of the AST
    // spelling: one program may instantiate same-named generics from
    // different modules, and the AST spelling would collide.
    std::string symbolBase;
    // Defining source module for imported generics (null for main-module
    // ones): specialization bodies must emit under the DEFINING module's
    // scope, not the use site's.
    const EmitOptions::SourceModule *source = nullptr;
    llvm::DenseMap<mlir::Type, std::string> specializations;
  };
  std::optional<std::pair<std::string, py::CallableType>>
  ensureGenericSpecialization(const parser::Node &anchor,
                              GenericFunctionInfo &generic,
                              py::CallableType target);
  // Argument specialization, and the record is the same GenericFunctionInfo
  // because the mechanism is: one extra body per demanded ground signature,
  // memoized on it, emitted under the defining module's scope. What differs
  // is what makes a signature ground -- there a solved type parameter, here
  // an argument standing a rung BELOW the declared parameter in the numeric
  // tower (`def f(x: float)` reached by `f(3)`).
  //
  // ⭐ Why a second body and not a conversion at the boundary: measured
  // against python3.14, converting is a wrong answer. `def p(x: float)`
  // called `p(3)` prints 3, not 3.0, and `def q(n: int)` called `q(True)`
  // prints True, not 1 -- the annotation is inert at a parameter and the
  // argument keeps its own type. The paths that already work agree: an
  // INLINED method (`C().m(3)` prints 6, `C().m(3.0)` prints 6.0) and a local
  // annotated binding (`x: float = 3; print(x)` prints 3) both specialize and
  // neither converts. A free function is the only spelling that fails,
  // because it is the only one whose ABI comes from the annotation.
  llvm::StringMap<GenericFunctionInfo> monomorphicFunctions;
  GenericFunctionInfo *lookupMonomorphicFunction(llvm::StringRef name);
  void recordMonomorphicFunction(llvm::StringRef key,
                                 const parser::Node &function,
                                 const FunctionSignature &sig,
                                 llvm::StringRef symbolBase,
                                 const EmitOptions::SourceModule *source);
  // A cheap AST + inference filter, run BEFORE anything is emitted, so the
  // ordinary path stays untouched for every call that cannot specialize.
  bool mayArgumentSpecialize(const parser::Node &expr,
                             const GenericFunctionInfo &info);
  // Emits the call. Takes the callee already emitted, and dispatches to the
  // DECLARED symbol whenever the specialization turns out not to apply, so
  // that this is a drop-in for the ordinary dispatch rather than a branch the
  // caller has to unwind.
  Value emitArgumentSpecializedCall(const parser::Node &expr,
                                    const parser::Node &calleeNode,
                                    GenericFunctionInfo &info,
                                    Value declaredCallee);
  // Runs `body` under the environment of the module that DEFINES an imported
  // generic, not the use site's. Scope ISOLATION rather than a plain push is
  // what keeps that honest: a plain push would let an unbound name in the
  // imported body silently resolve to a use-site local instead of being
  // diagnosed. Diagnostics raised inside are attributed to that module's
  // file.
  void emitInDefiningModuleScope(const EmitOptions::SourceModule &source,
                                 llvm::function_ref<void()> body);
  // Generic lookup for a callee spelling: the local registration first, then
  // the canonical import binding (imported generics register under their
  // canonical "<module>.<name>" symbol).
  GenericFunctionInfo *lookupGenericFunction(llvm::StringRef name);
  // Source module that defines class `className` ("<module>.<Class>"
  // contract names), null for main-module and manifest classes.
  const EmitOptions::SourceModule *
  sourceModuleForClass(llvm::StringRef className) const;
  Value emitGenericCall(const parser::Node &expr,
                        const parser::Node &calleeNode,
                        GenericFunctionInfo &generic);
  CallOperands emitCallOperands(const parser::Node &expr,
                                llvm::ArrayRef<Value> leadingPositional = {},
                                bool includeAstArguments = true,
                                py::CallableType expectedContract = {});
  Value emitCallableDispatch(const parser::Node &anchor, Value callee,
                             const CallOperands &operands,
                             mlir::Type resultOverride = {});
  // A manifest export is C behind its declared parameter, so an argument on a
  // lower numeric rung converts there (EmitterCalls.cpp). Null when nothing
  // needs widening; the returned node owns the rewritten arguments.
  parser::NodePtr
  widenNumericArgumentsForManifestCall(const parser::Node &expr,
                                       llvm::StringRef binding,
                                       py::CallableType declared);
  std::optional<Value> tryEmitIsInstanceCall(const parser::Node &expr,
                                             const parser::Node *calleeNode);
  // int(s, base): a synthesized module function, memoized by symbol. The
  // parse is ordinary Python over the str and int surface that already
  // compiles, so there is no new native for it.
  std::optional<Value> tryEmitIntBaseCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::string intBaseHelperSymbol;
  py::CallableType intBaseHelperCallable;
  std::optional<Value> tryEmitIntCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value> tryEmitFloatCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitPowCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  Value emitFloatFromInt(const parser::Node &anchor, Value argument);
  Value emitIntFromBool(const parser::Node &anchor, Value argument);
  std::optional<Value> tryEmitStrCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value>
  tryEmitContainerConstructorCall(const parser::Node &expr,
                                  const parser::Node *calleeNode);
  Value emitConstructorComprehension(const parser::Node &expr,
                                     parser::NodePtr argNode,
                                     llvm::StringRef which);
  std::optional<Value> emitListToTupleFreeze(const parser::Node &expr,
                                             const parser::Node &calleeNode,
                                             Value listValue);
  std::optional<Value> tryEmitListCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitPrintCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitBoolCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitAsciiCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitIssubclassCall(const parser::Node &expr,
                                             const parser::Node *calleeNode);
  std::optional<Value> tryEmitReducerCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::optional<Value> tryEmitLenCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value> tryEmitNextCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitRoundCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitHashCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  void emitSliceMutation(const parser::Node &target,
                         const parser::Node *containerNode,
                         const parser::Node &sliceNode,
                         llvm::StringRef methodName,
                         std::optional<Value> payload);
  std::optional<Value> tryEmitReprCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitFormatCall(const parser::Node &expr,
                                         const parser::Node *calleeNode);
  // f-string machinery: JoinedStr folds stringified pieces with str.__add__;
  // FormattedValue applies !r/!s/!a and dispatches __format__ statically.
  Value emitJoinedStr(const parser::Node &expr);
  Value emitFormattedValue(const parser::Node &expr);
  // str(value) semantics on an already-emitted value (str kept, source-class
  // __str__ inlined, manifest __str__, then __repr__) — print's stringify
  // for values instead of AST nodes.
  std::optional<Value> emitStringifyValue(const parser::Node &anchor,
                                          Value value);
  // !r / !s / !a conversion of an already-emitted value to str.
  std::optional<Value> emitConversionValue(const parser::Node &anchor,
                                           Value value, int64_t conversion);
  // format(value, spec) core: source-class __format__ first, manifest
  // __format__ second, str()-fallback for an absent/empty spec (CPython's
  // object.__format__ rule), diagnostic otherwise.
  Value emitFormatValue(const parser::Node &anchor, Value value,
                        std::optional<Value> spec, bool specKnownEmpty);
  Value emitEmptyStrConstant(const parser::Node &anchor);
  Value emitStrLiteralPiece(const parser::Node &anchor, llvm::StringRef text);
  bool canStringifyType(mlir::Type type);
  Value emitUnionStringify(const parser::Node &anchor, Value value,
                           py::UnionType unionType, unsigned index);
  // str.format with a compile-time template: fields are matched against the
  // call arguments during emission (R4: literal templates are checked here).
  std::optional<Value> tryEmitStrFormatCall(const parser::Node &expr,
                                            const parser::Node *calleeNode);
  // printf-style % formatting on a str left operand.
  Value emitPercentFormat(const parser::Node &expr);
  // Attribute/index accessors inside str.format replacement fields
  // ({0.attr} / {0[i]}), applied to an already-emitted value.
  std::optional<Value> emitValueAttribute(const parser::Node &anchor,
                                          Value object, llvm::StringRef attr);
  std::optional<Value> emitValueIndex(const parser::Node &anchor, Value object,
                                      llvm::StringRef indexText);
  std::optional<Value> rejectStubSourceCall(const parser::Node &expr,
                                            llvm::StringRef symbol,
                                            bool instantiation);
  // Taxonomy exceptions and user exception classes: the only contracts whose
  // manifest __str__ (the message) differs from __repr__ (ClassName(...)).
  bool isExceptionContractType(mlir::Type type) const;
  bool methodBindingBindsReceiver(const MethodBinding &method) const;
  Value emitDescriptorReceiver(const parser::Node &anchor, Value receiver,
                               const MethodBinding &method);
  Value emitMethodObject(const parser::Node &anchor, Value receiver,
                         const MethodBinding &method);
  // Operator protocol on a source-class receiver (x[i], x[i]=v, len(x),
  // i in x): the class method inlines with pre-built arguments, the same
  // dispatch as an explicit x.__getitem__(i) call.
  // A proved fact (`x is not None`, `isinstance(x, C)`) becoming a narrower
  // SSA value for the branch that proved it.
  void applyBranchNarrowing(const parser::Node &anchor,
                            const struct BranchTypeNarrowing &fact,
                            bool conditionIsTrue);
  Value emitInlineOperatorCall(const parser::Node &anchor, Value receiver,
                               const MethodBinding &method,
                               llvm::ArrayRef<Value> positional);
  // ⭐ THE gate. Every operator, every builtin whose job is to call a dunder,
  // and every protocol the emitter drives asks the receiver's SOURCE class
  // first, and asks it here. py ops resolve their target against the runtime
  // manifest, where a source class is not, so a site that forgets to ask
  // reaches the lowering as "runtime manifest has no C.__x__ method" -- which
  // is how __iter__, the unary dunders, abs/int/float/round/reversed,
  // __call__ and __divmod__ each arrived as a separate defect.
  //
  // nullopt means "no source class provides it": the caller takes its
  // manifest path. A value means the question is ANSWERED -- including the
  // refusal, because a receiver whose subclass overrides the method cannot be
  // dispatched from its static type and that is the answer.
  // `refused` is for a caller in STATEMENT position, which has nothing to
  // return: without it, `for x in b` over a subclass-overridden __iter__
  // printed the refusal and then a second complaint about __next__ on the
  // None the refusal handed back.
  std::optional<Value> tryEmitClassDunder(const parser::Node &anchor,
                                          Value receiver,
                                          llvm::StringRef dunder,
                                          llvm::ArrayRef<Value> positional = {},
                                          bool *refused = nullptr);
  // The same gate for the one dunder whose arguments are still AST: __call__
  // takes the call node's own arguments, keywords and defaults included, so
  // it cannot pre-build the operand list the other entry point takes.
  std::optional<Value> tryEmitClassDunderCall(const parser::Node &call,
                                              Value receiver,
                                              llvm::StringRef dunder);
  // Shared prologue of the two entry points above. nullopt with `refused`
  // set means the diagnostic is already out and the caller must not fall
  // through to its manifest path.
  std::optional<MethodBinding> resolveClassDunder(const parser::Node &anchor,
                                                  Value receiver,
                                                  llvm::StringRef dunder,
                                                  bool &refused);
  Value emitInlineMethodCall(const parser::Node &expr, Value receiver,
                             const MethodBinding &method);
  Value emitInlineMethodBody(const parser::Node &anchor, Value receiver,
                             bool bindDescriptorReceiver,
                             const MethodBinding &method,
                             llvm::ArrayRef<Value> positional,
                             const llvm::StringMap<Value> &keywords);
  Value emitClassInstantiation(const parser::Node &expr, llvm::StringRef name,
                               mlir::Type instanceType);
  // `super().m(...)` / `super(C, obj).m(...)`: compile-time expansion to the
  // MRO-next provider's method, inlined with the current receiver. Returns
  // nullopt when the call is not super-shaped.
  std::optional<Value> tryEmitSuperCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  // super().__init__(...) resolving onto a builtin exception base: binds the
  // runtime exception message of an exception-backed instance.
  Value emitSuperExceptionInit(const parser::Node &expr, Value receiver,
                               llvm::StringRef baseContract);
  // Unknown decorators are rejected (never silently ignored). The role
  // selects the recognized set; propertyNames lets method checks accept
  // `<prop>.setter` only for a property declared in the same class body.
  enum class DecoratorRole { Function, Method, Class };
  void checkDecorators(const parser::Node &node, DecoratorRole role,
                       const llvm::StringSet<> *propertyNames = nullptr);
  Value emitUnary(const parser::Node &expr);
  Value emitBinary(const parser::Node &expr);
  // Complex arithmetic (R6 value type): folds constant operands, promotes
  // constant real operands to complex, and dispatches +,-,*,/ through the
  // complex manifest methods. Returns nullopt when neither operand is
  // complex.
  std::optional<Value> emitComplexBinary(const parser::Node &expr, Value lhs,
                                         Value rhs, const parser::Node *op);
  Value emitCompare(const parser::Node &expr);
  // Scalar (non-Optional) comparison dispatch: primitive path, bool-vs-bool
  // truth compare, None-identity narrowing, membership, then the manifest
  // rich-comparison special ops. Shared by emitCompare and the Optional
  // member branch.
  Value emitScalarCompare(const parser::Node &expr, Value lhs, Value rhs,
                          const parser::Node *op);
  // `Optional[T] ==/!= x` (and the commuted form): dispatches on the union's
  // active member — a None member compares unequal to any concrete value, a
  // present member re-enters emitScalarCompare. Returns nullopt when neither
  // operand is an `Optional` of a single concrete member.
  std::optional<Value> emitOptionalCompare(const parser::Node &expr, Value lhs,
                                           Value rhs, const parser::Node *op);
  Value emitSubscript(const parser::Node &expr);
  // `a[i:j:k]` dispatches to the sequence's `__getslice__` manifest method
  // (start, stop, step, mask) — mask bit0/bit1 mark an explicit start/stop,
  // because the runtime defaults depend on the step's sign (R6: CPython
  // normalization, new copy).
  Value emitSliceSubscript(const parser::Node &expr, Value container,
                           const parser::Node &sliceNode);
  Value emitAttribute(const parser::Node &expr);
  Value emitAwait(const parser::Node &expr);
  Value emitAsyncioRunCall(const parser::Node &expr);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable,
                       const AwaitInferenceResult &inference);
  Value emitContainerLiteral(const parser::Node &expr,
                             mlir::Type expected = {});
  Value emitSetLiteral(const parser::Node &expr, mlir::Type expected = {});
  // `and`/`or` over non-bool operands: CPython's operand-value result,
  // restricted to operand combinations whose join is statically representable
  // (R1). The all-bool fast path stays in emitExpr.
  Value emitBoolOpValue(const parser::Node &expr, bool isAnd,
                        const std::vector<parser::NodePtr> &operands);
  Value emitListComp(const parser::Node &expr);
  Value emitDictComp(const parser::Node &expr);
  Value emitComprehension(const parser::Node &expr, bool isDict,
                          bool isSet = false);
  Value emitBindingRef(const parser::Node &anchor, llvm::StringRef binding,
                       mlir::Type type, llvm::ArrayRef<Value> captures = {});
  std::optional<Value> emitManifestFloatConstant(const parser::Node &anchor,
                                                 llvm::StringRef binding);
  std::optional<Value> emitManifestIntConstant(const parser::Node &anchor,
                                               llvm::StringRef binding);
  std::optional<Value> emitManifestStrConstant(const parser::Node &anchor,
                                               llvm::StringRef binding);
  std::optional<Value> emitStaticStringConstant(const parser::Node &anchor,
                                                llvm::StringRef binding,
                                                bool allowCallable = false);
  std::optional<Value> emitStaticIntConstant(const parser::Node &anchor,
                                             llvm::StringRef binding);
  std::optional<Value> emitLiteralTypeConstant(const parser::Node &anchor,
                                               mlir::Type type);
  Value emitFunctionObject(const parser::Node &anchor,
                           llvm::StringRef symbolName, mlir::Type type,
                           llvm::ArrayRef<Capture> captures);
  // R6 nonlocal cells: a boxed local's storage is an instance of a
  // synthesized one-field class ("__ly_cell$N" with field "v").
  static bool isCellContract(mlir::Type type);
  mlir::Type cellContentType(mlir::Type cellType);
  mlir::Type ensureCellClass(mlir::Type contentType,
                             const parser::Node &anchor);
  Value emitCellAlloc(const parser::Node &anchor, Value initial);
  Value emitCellLoad(const parser::Node &anchor, const Value &cell);
  void emitCellStore(const parser::Node &anchor, const Value &cell,
                     Value value);
  std::optional<Value>
  emitPrimitiveConstructorCall(const parser::Node &expr,
                               const parser::Node *calleeNode);
  std::optional<Value> emitPrimitiveFactoryCall(const parser::Node &expr,
                                                const parser::Node *calleeNode);
  std::optional<Value> emitPrimitiveRuntimeCall(const parser::Node &expr,
                                                const parser::Node *calleeNode);
  Value emitToPrimCall(const parser::Node &expr);
  std::optional<Value>
  emitDirectPrimitiveFunctionCall(const parser::Node &expr,
                                  const parser::Node *calleeNode);
  llvm::SmallVector<mlir::Value, 4>
  emitPrimitiveTensorIndices(const parser::Node &expr,
                             mlir::RankedTensorType tensorType,
                             const parser::Node *slice);
  std::optional<Value> emitPrimitiveTensorGetItem(const parser::Node &expr,
                                                  Value container,
                                                  const parser::Node *slice);
  std::optional<Value> emitPrimitiveTensorSetItem(const parser::Node &expr,
                                                  Value container,
                                                  const parser::Node *slice,
                                                  Value element);
  std::optional<Value> emitPrimitiveBinary(const parser::Node &expr, Value lhs,
                                           Value rhs, const parser::Node *op);
  std::optional<Value> emitPrimitiveCompare(const parser::Node &expr, Value lhs,
                                            Value rhs, const parser::Node *op);
  Value emitPrimitiveConstant(const parser::Node &anchor,
                              const PrimitiveConstant &constant);
  Value coercePrimitiveInteger(Value value, mlir::IntegerType targetType,
                               const parser::Node &anchor);
  // Adapt an already-emitted value to a primitive scalar type: primitive
  // scalars coerce by width, Python int/float values unbox through
  // py.cast.to_prim. Null value on failure (with a diagnostic).
  mlir::Value coerceToPrimitiveScalar(Value value, mlir::Type elementType,
                                      const parser::Node &anchor);
  // Emit one element of a shaped-primitive constructor: numeric literals fold
  // to constants, everything else goes through coerceToPrimitiveScalar.
  mlir::Value emitPrimitiveElementValue(const parser::Node *node,
                                        mlir::Type elementType,
                                        const parser::Node &anchor);
  Value emitNone(const parser::Node &anchor);
  Value emitPack(mlir::ArrayRef<Value> values,
                 llvm::ArrayRef<char> unpacked = {});
  Value coerceValue(Value value, mlir::Type targetType,
                    const parser::Node &anchor);
  // bool, int and float: the three contracts that carry a VALUE rather than an
  // object handle, and so the three between which a retyping is a lie.
  bool isNumericPrimitiveContract(mlir::Type type) const;
  mlir::Value emitBoolValue(Value value, const parser::Node &anchor);

  template <typename Op>
  Value emitBinarySpecial(const parser::Node &anchor, llvm::StringRef method,
                          Value lhs, Value rhs, mlir::Type resultType);
  template <typename Op>
  Value emitUnarySpecial(const parser::Node &anchor, llvm::StringRef method,
                         Value input, mlir::Type resultType);

  mlir::ModuleOp module;
  const parser::Node &moduleNode;
  mlir::MLIRContext &context;
  std::string moduleName;
  std::string sourceName;
  std::string activePackageName;
  EmitOptions options;
  mlir::OpBuilder builder;
  TypeSystem types;
  llvm::StringMap<GenericFunctionInfo> genericFunctions;
  llvm::StringMap<GenericClassInfo> genericClasses;
  // Spellings the MAIN module's own top-level `def` / `class` statements
  // bind, collected before anything is typed or emitted. Declaration order is
  // deliberately not modelled: registerModule binds every top-level signature
  // up front so forward references and mutual recursion resolve, so a
  // top-level name means the same thing at every point of the module.
  llvm::StringSet<> moduleFunctionNames;
  llvm::StringSet<> moduleClassNames;
  // Main-module top-level `def`s whose spelling is also a manifest builtin's
  // binding name (`def len`), mapped to the symbol they are emitted under.
  // py.binding_ref carries a NAME, and the runtime lowering resolves that name
  // against the manifest before it looks for a user func.func, so emitting the
  // user function as @len would hand every reference to the builtin. Renaming
  // the emitted symbol keeps the two entities distinguishable at the only
  // layer that can tell them apart -- the one that knows which is which.
  llvm::StringMap<std::string> shadowedBuiltinSymbols;
  // Solved type arguments per specialized class contract, in parameter order.
  llvm::StringMap<llvm::SmallVector<std::pair<std::string, mlir::Type>, 4>>
      classTypeArguments;
  // Instantiations whose class contract is allocated but whose body has not
  // been emitted. The queue exists because the FIRST demand for a
  // specialization arrives during registerModule's signature fixpoint (a
  // parameter annotated `C[int]`), where no class may be emitted yet: the
  // top-level declarations are not established and the fixpoint reruns.
  // Allocating the contract name there and draining at the declaration phase
  // keeps the signature memo consistent with what is eventually emitted.
  struct PendingClassSpecialization {
    std::string base;
    std::string symbol;
    llvm::SmallVector<mlir::Type, 4> arguments;
  };
  std::vector<PendingClassSpecialization> pendingClassSpecializations;
  // Set once the declaration phase begins: from then on a newly demanded
  // instantiation emits immediately, because a use site needs its class
  // facts (fields, methods, MRO) before the statement it appears in
  // finishes.
  bool genericClassEmissionReady = false;
  parser::Diagnostics diagnostics;
  llvm::StringMap<Value> values;
  llvm::StringMap<PrimitiveConstant> primitiveConstants;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classFieldBindings;
  // Declaration order of each class's fields (classFieldBindings is
  // unordered); drives the synthesized field-record constructor.
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classFieldOrders;
  // Classes written as `class P(NamedTuple)`: their instances are tuples, so
  // a literal subscript folds to the field at that position (EmitterExpressions).
  llvm::StringSet<> namedTupleContracts;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classStaticAttrBindings;
  llvm::StringMap<llvm::StringMap<MethodBinding>> classMethodBindings;
  // Canonical (resolved) base contract names per class, in declaration order.
  llvm::StringMap<llvm::SmallVector<std::string, 4>> classBaseNames;
  // C3 linearization per class (self first, canonical contract names).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classMros;
  // The module's class hierarchy as WRITTEN, filled by collectTopLevelBindings
  // before any emission, so that "does a subclass override this" does not
  // depend on where in the file it is asked.
  llvm::StringMap<llvm::SmallVector<std::string, 4>> declaredClassBases;
  llvm::StringMap<llvm::StringSet<>> declaredClassMethods;
  llvm::StringMap<llvm::StringSet<>> declaredClassAttributes;
  // How many except handler bodies enclose the statement being emitted. A bare
  // `raise` re-raises what a handler caught, so at zero there is nothing to
  // re-raise -- the question the lowering cannot ask, because `py.try`'s
  // regions are gone by then.
  unsigned exceptHandlerDepth = 0;
  // Fields declared by the class body itself (classFieldOrders holds the
  // MRO-merged instance layout).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classOwnFieldOrders;
  // MRO-merged class attribute declaration order and initializer expression
  // per class (classStaticAttrBindings holds the merged types).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classStaticAttrOrders;
  llvm::StringMap<llvm::StringMap<mlir::Attribute>> classStaticAttrValues;
  // Dataclass field default expressions (AnnAssign values), per class; MRO
  // walks reuse a base dataclass's defaults for inherited fields.
  llvm::StringMap<llvm::StringMap<parser::NodePtr>> classFieldDefaultNodes;
  // Mutable (slot-backed) class attributes, keyed by the DEFINING class:
  // attribute -> widened storage type. They live in module-global object
  // cells named "<class>.<attr>", initialized at the class statement's
  // position in module flow; subclass reads resolve to the defining class's
  // cell along the MRO (CPython shares the base's attribute until a write
  // creates a subclass shadow -- shadow-creating writes are diagnosed).
  llvm::StringMap<llvm::StringMap<mlir::Type>> classAttrSlots;
  // Synthesized dataclass method ASTs (__init__/__repr__/__eq__): owned here
  // because the parse tree does not contain them.
  std::vector<parser::NodePtr> synthesizedClassMethods;
  // Module-level mutable globals, opted in by an int annotation at module
  // scope (`NAME: int = ...`). Backed by process-lifetime storage so reads
  // are async-signal-safe (see py.global.get/set); referenced from any scope,
  // written from module scope or a `global NAME` declaration in a function.
  llvm::StringMap<mlir::Type> moduleGlobals;
  // Module-scope names bound exactly once to a literal: not cells, but their
  // references re-emit the literal, so a function body can read them
  // (collectModuleGlobals). The node is owned by the parse tree.
  llvm::StringMap<const parser::Node *> moduleConstantBindings;

  // ⭐ The three module-scope VALUE bindings above, hidden for the duration of
  // a walk that emits ANOTHER module's code. `TypeSystem::ScopeIsolation` does
  // the same for the type scopes and these are the rest of it: one
  // `ModuleEmitter` emits the program and every module it imports, so a name
  // registered here is otherwise in scope inside every stdlib body.
  //
  // ⛔ What that cost, since "the importer's names are merely also visible"
  // sounds harmless: they SHADOW at the type level. `iterdir` has a local
  // `names`, and a program with any annotated `names` global failed inside
  // pathlib -- "'builtins.int' does not provide manifest method 'sort'",
  // reported against a file the program never wrote, naming neither the
  // global nor the collision.
  //
  // Python resolves a function's globals in the module that DEFINES it, so
  // there is nothing here to reconcile: the importer's module scope simply is
  // not in scope. Restored on the way out, because the importer's own bodies
  // are emitted after.
  class ImporterModuleScope {
  public:
    explicit ImporterModuleScope(ModuleEmitter &emitter)
        : emitter(&emitter), globals(std::move(emitter.moduleGlobals)),
          constants(std::move(emitter.moduleConstantBindings)),
          primitives(std::move(emitter.primitiveConstants)) {
      emitter.moduleGlobals.clear();
      emitter.moduleConstantBindings.clear();
      emitter.primitiveConstants.clear();
    }
    ImporterModuleScope(const ImporterModuleScope &) = delete;
    ImporterModuleScope &operator=(const ImporterModuleScope &) = delete;
    ~ImporterModuleScope() {
      emitter->moduleGlobals = std::move(globals);
      emitter->moduleConstantBindings = std::move(constants);
      emitter->primitiveConstants = std::move(primitives);
    }

  private:
    ModuleEmitter *emitter;
    llvm::StringMap<mlir::Type> globals;
    llvm::StringMap<const parser::Node *> constants;
    llvm::StringMap<PrimitiveConstant> primitives;
  };
  // Names whose flow type a branch narrowing replaced, and what it was before.
  // A WRITE inside the branch is not constrained by the narrowing -- `xs = []`
  // under `if xs is None:` wants the declared `list[int] | None`, not the
  // `None` the read side sees (EmitterControlFlow.cpp / the Assign rule).
  llvm::StringMap<mlir::Type> narrowedFromTypes;
  // Names declared `global` in the function currently being emitted (writes
  // to them target the module global instead of a new local). Saved/restored
  // around each callable body.
  llvm::StringSet<> currentGlobalDecls;
  // Locals of the function currently being emitted that some nested function
  // declares `nonlocal` (R6): their storage is a shared refcounted cell (a
  // synthesized one-field class instance), created at the first binding.
  // Every read/write in every scope goes through the cell's field, so the
  // enclosing scope and its closures observe one mutable slot. Saved/restored
  // around each callable body.
  llvm::StringSet<> currentBoxedLocals;
  bool atModuleScope = false;
  mlir::Type currentReturnType;
  mlir::Type currentGeneratorSendType;
  std::string currentFunctionPrefix;
  std::vector<parser::NodePtr> synthesizedDefaultProviders;
  // Non-constant defaults of MODULE-level defs (R6): evaluated once when
  // __main__ reaches the def statement and parked in a module-lifetime
  // object-global cell; call sites read the cell instead of re-evaluating.
  struct PendingDefaultCell {
    std::string cellName;
    parser::NodePtr expr;
    mlir::Type declaredType;
  };
  llvm::DenseMap<const parser::Node *,
                 llvm::SmallVector<PendingDefaultCell, 2>>
      pendingDefaultCells;
  // Non-constant defaults of NESTED defs: evaluated once when the enclosing
  // execution reaches the def statement, carried as synthetic closure
  // captures (slot -> capture index) shared by every call in that execution.
  llvm::DenseMap<const parser::Node *,
                 llvm::SmallVector<std::pair<unsigned, unsigned>, 2>>
      nestedDefaultCaptures;
  unsigned syntheticFunctionCounter = 0;
  unsigned listCompCounter = 0;
  // Cell classes are synthesized once per (widened) content type.
  llvm::DenseMap<mlir::Type, mlir::Type> cellClassContracts;
  unsigned cellClassCounter = 0;
  llvm::SmallVector<WithCleanup, 8> activeWithCleanups;
  // The `with` items whose enters have been scheduled but not yet emitted:
  // a synthesized LyWithEnter statement names one by index, because the try
  // that guards it has to be opened before it runs.
  struct PendingWithItem {
    const parser::Node *item = nullptr;
    bool async = false;
  };
  llvm::SmallVector<PendingWithItem, 4> pendingWithItems;
  // Values a desugaring emitted once and refers to through a synthesized
  // `LyValueRef` node, so the two places that need the subexpression share
  // one evaluation (EmitterExpressions.cpp).
  llvm::SmallVector<Value, 4> pendingValueRefs;
  llvm::SmallVector<InlineReturnContext, 4> inlineReturnContexts;
  llvm::SmallVector<LoopControlContext, 4> loopControlContexts;
  // Innermost = the class method body currently being emitted (inline or
  // standalone); zero-argument super() reads the defining class and the
  // receiver parameter name from here.
  struct SuperContext {
    std::string definingClass;
    std::string selfName;
  };
  llvm::SmallVector<SuperContext, 4> superContexts;
  // Method bodies currently being inlined, innermost last. Inlining emits
  // BOTH arms of every branch, so a cycle in the inline graph has no base
  // case and expands forever: re-entering a body already on this stack is
  // always unbounded, never a recursion that would have terminated.
  llvm::SmallVector<const parser::Node *, 8> methodsBeingInlined;
};

} // namespace lython::emitter
