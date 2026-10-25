#pragma once

#include "Emitter.h"
#include "EmitterState.h"
#include "EmptyLiteralSeed.h"
#include "TypeSystem.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <set>
#include <cstddef>
#include <map>

namespace lython::emitter {

// The name the traceback module is bound under when the program needs it for
// itself -- a finalizer's report, `sys.exc_info()`'s traceback.
inline constexpr llvm::StringLiteral kTracebackModuleBinding{"__ly_traceback"};

// What a stub's contracts make of what TypeScript left untyped (`t.Any`):
// left alone, or a type for each direction a value crosses.
struct StubContractPolicy {
  mlir::Type anyResult;
  mlir::Type anyParameter;
  // A static method is also a method of the class's VALUES: a JavaScript
  // static is called with the constructor as `this`, and the stub's
  // `X.new(...)` is how `new X(...)` is spelled.
  bool staticMethodsTakeTheValue = false;
  // A base every class gets ahead of `object`: for `js`, `_js.JsProxy`, which
  // is what each of its values is at run time and what tells a host class
  // from a program's own class of the same name.
  std::string commonBase;
  // A parameter of this type -- `js.Function` -- also takes any Python
  // callable, which crosses as a function the host can call.
  mlir::Type callableParameter;
};

// The JavaScript host's `js` module: the stub the compiler ships, not a file
// of that name beside the program.
bool isJsHostModule(const EmitOptions::SourceModule &source);

// asyncio's namespace under a name the program cannot spell, for the `await`
// on a host Promise the emitter turns into a call into it.
inline constexpr llvm::StringLiteral kAsyncioName = "__ly_asyncio$";

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
                             const YieldFromInferenceResult &inference);
  void predeclareTopLevel();
  void predeclareSourceModules();
  void declareJsHostModule();
  FunctionSignature
  importedFunctionSignature(const EmitOptions::SourceModule &source,
                            const std::vector<parser::NodePtr> &body,
                            const parser::Node &function);
  llvm::StringSet<> importedSignatureScopes;
  // ⛔ Not recomputed at every binding: a signature binds its module's imports,
  // and binding an import reads the signature of every def that module
  // exports, so the work multiplied with each level of the import graph (a
  // chain of 12 modules took 22 s, CPython's unittest package never finished).
  // ⛔ And not kept for the whole compile: a signature read before the class
  // declarations exist walks its body against no class schemas, so the
  // memo is dropped at every step that declares more (forgetImportedSignatures).
  // Keyed with whether defaults describe parameters, which is false only
  // during the module fixpoint and changes what a body walk answers.
  llvm::DenseMap<std::pair<const parser::Node *, unsigned>, FunctionSignature>
      importedSignatures;
  // The signatures `bindSourceModuleLocals` binds an imported module's defs
  // under. Every method body inlined from the module binds its scope again,
  // and reading every def's signature each time was half of the emitter on a
  // program that imports unittest. The answer is the same each time: every
  // caller isolates the scopes and binds the module's imports first, as the
  // memo above does, and what a reading does besides answering is recorded
  // once either way (annotation diagnostics are deduplicated, a generic class
  // is specialized once). Dropped with the imported ones, for the same reason.
  llvm::DenseMap<std::pair<const parser::Node *, unsigned>, FunctionSignature>
      sourceLocalSignatures;
  // The type aliases each imported module binds, read in its own scope, for
  // an importer that spells them through the module (`m.Pair`). Dropped with
  // the signatures: an alias of a generic class read before that class was
  // registered names the unspecialized class.
  std::map<std::string, std::vector<std::pair<std::string, mlir::Type>>>
      sourceModuleTypeAliasMemo;
  const std::vector<std::pair<std::string, mlir::Type>> &
  sourceModuleTypeAliases(const EmitOptions::SourceModule &source);
  // An imported module's top-level defs and classes by name, first one
  // first. `bindSourceModuleName` scanned the module's statements for each
  // name it bound, and every method body inlined from a module binds that
  // module's imports again. Keyed like `moduleConstantMemo`, by the
  // statements' first node and count, which outlive the compile's readings of
  // them.
  struct TopLevelDefinitions {
    llvm::StringMap<const parser::Node *> functions;
    llvm::StringMap<const parser::Node *> classes;
    // `alias = other` (moduleAliasTarget's answer), and the modules an
    // `import M as member` publishes under each member, in statement order.
    llvm::StringMap<std::string_view> aliasTargets;
    llvm::StringMap<llvm::SmallVector<std::string_view, 1>> importedMembers;
  };
  llvm::DenseMap<std::pair<const void *, unsigned>, TopLevelDefinitions>
      topLevelDefinitionMemo;
  const TopLevelDefinitions &
  topLevelDefinitions(const std::vector<parser::NodePtr> &body);
  // The first of `sourceModuleTypeAliases` under each name, for the reader
  // that asks about one name at a time: searching the list once per imported
  // name was the largest single line of the emitter on test_argparse.
  llvm::StringMap<llvm::StringMap<mlir::Type>> sourceModuleTypeAliasIndex;
  const llvm::StringMap<mlir::Type> &
  sourceModuleTypeAliasesByName(const EmitOptions::SourceModule &source);
  // What binding each imported module's imports wrote, replayed by the next
  // binding of the same module (`bindModuleImportScope`). Valid while no
  // signature memo was dropped and no module global was added since, the two
  // things a reading consults that change during emission.
  struct ImportScopeMemo {
    std::shared_ptr<TypeSystem::FrameRecord> record;
    unsigned generation = 0;
    std::size_t moduleGlobalCount = 0;
  };
  llvm::DenseMap<const parser::Node *, ImportScopeMemo> importScopeMemo;
  unsigned importScopeGeneration = 0;
  void bindModuleImportStatements(const parser::Node &sourceModule,
                                  bool diagnoseUnsupported);
  void forgetImportedSignatures() {
    ++importScopeGeneration;
    importedSignatures.clear();
    sourceLocalSignatures.clear();
    sourceModuleTypeAliasMemo.clear();
    sourceModuleTypeAliasIndex.clear();
  }
  // The host's global functions (Window's methods), and the spellings the
  // program imported them by.
  llvm::StringSet<> jsHostGlobalFunctions;
  llvm::StringMap<std::string> jsHostFunctionAliases;
  std::optional<Value> tryEmitJsHostFunctionCall(const parser::Node &expr,
                                                 llvm::StringRef callee);
  // A value of one of the host's `js` contracts (EmitterImports.cpp).
  bool isJsHostValueType(mlir::Type type) const;
  // A host member read or call whose declared type is a union: the host's
  // value is taken as it comes and dispatched to the member it is.
  Value adaptJsHostResult(const parser::Node &anchor, mlir::Operation *op,
                          Value declared);
  // A Python callable where a host member takes a callback (docs/js-host.md).
  Value wrapJsCallback(const parser::Node &anchor, py::CallableType declared,
                       Value callable);
  void wrapJsCallbackArguments(const parser::Node &anchor,
                               CallInferenceResult &inference,
                               CallOperands &operands);
  // One synthesized wrapper per (declared callback type, callable's type).
  llvm::DenseMap<std::pair<mlir::Type, mlir::Type>, std::string>
      jsCallbackWrappers;
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
                                llvm::StringRef localName,
                                unsigned namespaceDepth = 0);
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
  // A contract this compiler DECLARED -- written in this module or imported
  // from another -- as opposed to a manifest one.
  //
  // ⭐ NOT the name-shape test (`isSourceDefinedContract`, "the name has no
  // dot"). An imported class's contract name is `mod.Base`, which looks
  // exactly like `builtins.int` to that test, so every question asked with it
  // answered "manifest" for a program's own class the moment it crossed a
  // file. Ask the map of what was declared, which is filled for both.
  bool isSourceClassContract(mlir::Type type) const;
  // The second half of `collectTopLevelBindings`, run once the imports are
  // bound: neither a base spelled `module.Class` nor one spelled with the bare
  // name a `from module import Class` bound can be canonicalized before that.
  void resolveTopLevelBaseSpellings();
  // A module-level class whose base is a class of this module that no earlier
  // statement has bound yet.
  void refuseBasesBoundLater(const parser::Node &root);
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
  // Does any class in the program derive from this one? A classmethod's `cls`
  // binds the RUNTIME class, so a subclass that redeclares NOTHING still has
  // its own answer -- which is why the override test is not the question here.
  bool classHasDeclaredSubclass(llvm::StringRef receiverClass) const;
  bool methodBodyReadsFirstParameter(const parser::Node &method) const;
  bool subclassOverridesMethod(llvm::StringRef receiverClass,
                               llvm::StringRef methodName) const;
  // The same question about a class-level BINDING rather than a method
  // (EmitterClasses.cpp).
  // Whether `candidate` reaches a declaration of `name` that the receiver
  // class does not already resolve to. The gate below and the dispatcher's
  // candidate scan ask this same question; asking it two ways is what left a
  // program refused by one and unanswerable by the other.
  bool candidateRedeclares(const llvm::StringMap<llvm::StringSet<>> &declarations,
                           llvm::StringRef receiverClass,
                           llvm::StringRef candidate, llvm::StringRef name) const;
  bool subclassShadowsAttribute(llvm::StringRef receiverClass,
                                llvm::StringRef attributeName) const;
  bool subclassRedeclares(const llvm::StringMap<llvm::StringSet<>> &declarations,
                          llvm::StringRef receiverClass,
                          llvm::StringRef name) const;
  // Every ancestor of a class: the program's as written, and a runtime
  // base's own (EmitterClasses.cpp).
  void collectDeclaredAncestors(llvm::StringRef cls,
                                llvm::StringSet<> &out) const;
  // The ancestors above, per class, and the answers of `subclassRedeclares`
  // per table, class and name. Asked for every method call -- each answer
  // walked every class and rebuilt both ancestor sets -- they were a third of
  // the emitter on test_argparse. Dropped by every write to the tables they
  // read (`forgetDeclarationMemos`).
  mutable llvm::StringMap<llvm::StringSet<>> declaredAncestorMemo;
  mutable llvm::StringMap<bool> redeclarationMemo;
  const llvm::StringSet<> &declaredAncestors(llvm::StringRef cls) const;
  void forgetDeclarationMemos() {
    declaredAncestorMemo.clear();
    redeclarationMemo.clear();
  }
  // The one gate every method dispatch goes through: emits the diagnostic and
  // returns true when the receiver's static class cannot answer the call
  // (EmitterClasses.cpp).
  bool refuseUnresolvableDispatch(const parser::Node &anchor, Value receiver,
                                  llvm::StringRef methodName,
                                  const parser::Node *receiverNode = nullptr,
                                  bool throughSuper = false);
  // The same question without the diagnostic, for the sites that can answer
  // the call another way before refusing it (EmitterClasses.cpp).
  bool dispatchIsUnresolvable(Value receiver, llvm::StringRef methodName,
                              const parser::Node *receiverNode,
                              bool throughSuper) const;
  // A call the static receiver type cannot resolve, answered by a synthesized
  // module function that tests the runtime class and calls the body that class
  // declares (EmitterCalls.cpp). Nothing when the shape is outside what the
  // synthesis covers, and the caller then refuses as before.
  std::optional<Value> tryEmitVirtualDispatch(const parser::Node &expr,
                                              const parser::Node &calleeNode,
                                              const parser::Node *receiverNode,
                                              Value receiver,
                                              llvm::StringRef methodName);
  // The same dispatch for an operator site, whose operands are already values.
  std::optional<Value>
  tryEmitVirtualDispatchWithValues(const parser::Node &anchor, Value receiver,
                                   llvm::StringRef methodName,
                                   llvm::ArrayRef<Value> positional);
  // `u.method(...)` on a union receiver, by tag (EmitterCalls.cpp). Nothing --
  // and nothing emitted -- when a member cannot answer, and the caller then
  // refuses as before.
  std::optional<Value> tryEmitUnionMethodCall(const parser::Node &expr,
                                              Value receiver,
                                              llvm::StringRef methodName);
  struct VirtualDispatchHelper {
    std::string symbol;
    py::CallableType callable;
  };
  // The dispatcher for one (receiver class, method), synthesized on first use.
  // Null when the shape is outside what the synthesis covers.
  const VirtualDispatchHelper *virtualDispatcherFor(
      const parser::Node &anchor, Value receiver, llvm::StringRef methodName,
      unsigned argumentCount, bool asProperty = false,
      llvm::ArrayRef<std::string> keywordNames = {}, bool asAttribute = false,
      bool asSetter = false, bool asField = false);
  // `lhs <op> rhs` over two source-class values where the right one's runtime
  // class may strictly derive the left's and answer the reflected comparison
  // first. Null when no pair of the program's classes reaches that.
  const VirtualDispatchHelper *subclassFirstCompareFor(const parser::Node &anchor,
                                                       Value lhs, Value rhs,
                                                       llvm::StringRef opKind);
  // Non-zero while such a dispatcher's body is emitted: its arms are the
  // one-receiver comparisons.
  unsigned subclassFirstSuppressed = 0;
  void emitDispatcherBody(const parser::Node &def, const std::string &symbol,
                          const FunctionSignature &sig, bool suppressGate);
  // A base's instance FIELD read where a subclass makes the name readable
  // another way (a class attribute with a value, a property).
  std::optional<Value> tryEmitVirtualFieldRead(const parser::Node &anchor,
                                               Value receiver,
                                               llvm::StringRef fieldName,
                                               mlir::Type fieldType);
  bool subclassReadsFieldOtherwise(llvm::StringRef receiverClass,
                                   llvm::StringRef fieldName) const;
  // Whether any class in `cls`'s line, or any subclass of it, stores the
  // name on `self` -- whether an instance can ever hold it.
  bool fieldEverStored(llvm::StringRef cls, llvm::StringRef fieldName) const;
  // Whether `cls` or an ancestor binds the name by a plain class-level
  // assignment or a property (`declaredClassReadableNames`).
  bool classLineBindsReadable(llvm::StringRef cls,
                              llvm::StringRef name) const;
  // `self.kind` where a subclass redeclares the class attribute `kind`: the
  // same dispatcher, reading a class attribute instead of calling a method.
  std::optional<Value> tryEmitVirtualAttributeRead(const parser::Node &anchor,
                                                   Value receiver,
                                                   llvm::StringRef attrName);
  // The forwarding body a bound METHOD OBJECT needs when the receiver's class
  // has an overriding subclass: `def m(self, ...): return __lyvdisp$N(self,
  // ...)`. Null when no dispatcher covers the shape, in which case the method
  // object keeps the static body it always had.
  const parser::Node *virtualMethodObjectDef(const parser::Node &anchor,
                                             Value receiver,
                                             const MethodBinding &binding);
  // The same, for a `@staticmethod` read off a base-typed instance. A static
  // binding has no receiver to bind, so the wrapper INTRODUCES one and the
  // result is an ordinary instance-kind binding over it -- which is what makes
  // `m = x.s` carry the receiver the dispatcher tests.
  std::optional<MethodBinding>
  virtualStaticMethodObjectBinding(const parser::Node &anchor, Value receiver,
                                   const MethodBinding &binding);
  // The same dispatch for a `@property` READ through a base-typed receiver.
  std::optional<Value> tryEmitVirtualPropertyRead(const parser::Node &anchor,
                                                  Value receiver,
                                                  llvm::StringRef propertyName);
  // And the WRITE. `x.v = n` inlines the setter the receiver's STATIC class
  // resolves to, which is the base's body for a base-typed receiver -- the
  // same wrong answer the read had, reached through the assignment path.
  bool tryEmitVirtualPropertyWrite(const parser::Node &anchor, Value receiver,
                                   llvm::StringRef propertyName, Value value);
  // Whether `className`'s `propertyName` is a property that still HAS a
  // setter: a subclass redeclaring the getter replaces the whole descriptor,
  // so an inherited setter is not reachable through it.
  bool classPropertyHasSetter(llvm::StringRef className,
                              llvm::StringRef propertyName) const;
  // Non-zero while a property dispatcher's body is being emitted, where the
  // unresolvable-dispatch gate is the question the dispatcher answers.
  unsigned virtualPropertyBodyDepth = 0;
  // Every module's ClassDefs have been emitted, so a dispatcher's arms can
  // name any class's methods; before then its body is queued here.
  bool classDeclarationsEmitted = false;
  std::vector<std::function<void()>> deferredDispatcherBodies;
  // Keyed "<class>.<method>", so one dispatcher serves every call site and a
  // method that dispatches on itself terminates. Filled BEFORE the body is
  // emitted for that second reason.
  llvm::StringMap<VirtualDispatchHelper> virtualDispatchHelpers;
  // A dispatcher's last arm for an exception's text: the `str(__ly_recv)` or
  // `repr(__ly_recv)` call it synthesized, which the builtin's own text
  // answers from the static type (virtualDispatcherFor).
  llvm::DenseSet<const parser::Node *> builtinTextFallbacks;
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
  void emitClassAttrInitializers(const parser::Node &classDef,
                                 llvm::StringRef contractName = {});
  // ⭐ AN IMPORTED CLASS'S ATTRIBUTE INITIALIZERS HAVE NO POSITION IN MODULE
  // FLOW, because an imported module's body does not run -- so they are queued
  // as its classes are declared and emitted at the start of `__main__`, in
  // import order, which is where CPython runs them.
  struct PendingClassAttrInit {
    const parser::Node *classDef = nullptr;
    std::string contractName;
    const EmitOptions::SourceModule *source = nullptr;
  };
  llvm::SmallVector<PendingClassAttrInit, 4> importedClassAttrInits;
  // The module globals, default cells and class attributes of every imported
  // module, each module's in the order its statements are written.
  void emitImportedModuleInitializers();
  bool statementHasImportedDefaultCells(const parser::Node &statement);
  // ⭐ AND THE SAME QUESTION ONE LEVEL UP: a module-level CONTAINER constant
  // in an imported module. A constant travels as a literal TYPE, and a
  // container has no literal spelling, so `NAMES = ["a"]` did not resolve from
  // the importer ("module 'm' has no attribute 'NAMES'") nor from a function
  // in its own module ("unresolved name 'NAMES'"). A module global is what the
  // main module gives one; these are the same storage, initialized at the
  // start of `__main__` in import order.
  struct PendingModuleGlobalInit {
    const parser::Node *value = nullptr;
    std::string globalName;
    const EmitOptions::SourceModule *source = nullptr;
    // The def a DECORATOR application is about, if any. While the decoration
    // runs, that name must read the emitted SYMBOL and not the cell it is
    // about to fill -- the same rule the main module's decorator pass keeps.
    std::string decoratorSubject = {};
  };
  llvm::SmallVector<PendingModuleGlobalInit, 4> importedModuleGlobalInits;
  void collectImportedModuleGlobals();
  // `sys.stdout` / `sys.stderr` that some module the program compiles
  // assigns: each becomes the module global of its name, holding either text
  // stream the runtime has (EmitterGlobals.cpp). Found before anything is
  // bound, filled at the start of `__main__`.
  llvm::SmallVector<std::string, 2> replacedStreams;
  void collectReplacedStreams();
  void emitReplacedStreamInitializers(const parser::Node &anchor);
  // The replaced stream an assignment target names, if it names one.
  std::optional<std::string>
  replacedStreamTarget(const parser::Node &target) const;
  // The `mod.NAME` global an imported module's constant lives in, or empty.
  std::string importedModuleGlobalFor(llvm::StringRef binding) const;
  // CPython calls a base's `__init_subclass__` when a subclass is DEFINED.
  // Emitted at the class statement's position in module flow, beside the
  // attribute initializers, for the same reason.
  void emitInitSubclassHook(const parser::Node &classDef,
                            llvm::StringRef contractName = {});
  // A cell, empty and binding-tracked, for every local of `callable` whose
  // first binding comes after the nested def or lambda that reads it. The
  // assignment path only makes a cell at a name's FIRST binding, so without
  // this the reader has nothing to capture and the name is dropped.
  void emitForwardBoundCells(const parser::Node &callable);
  // Module-level names bound to a type expression with no runtime value (a
  // subscript or a `|` union, or a name standing for one). Their assignment is
  // not emitted, so a later alias that spells one must not be either.
  llvm::StringSet<> valuelessTypeAliases;
  // Module-level names bound ONCE to a bare builtin spelling (`Text = str`).
  // A call through one is that builtin's call, which is keyed on the spelling.
  llvm::StringMap<std::string> builtinValueAliases;
  Value emitNestedFunctionDecl(const parser::Node &function);
  mlir::ArrayAttr emitCallableDefaultValues(const parser::Node &function,
                                            const FunctionSignature &sig,
                                            llvm::StringRef symbolName);
  Value emitLambda(const parser::Node &expr, py::CallableType expected = {});
  // `source` names the imported module a class was declared in, and is what
  // says its attribute initializers will be RUN (queued for the start of
  // `__main__`). A generic specialization also carries a symbolName and has no
  // such position, so it keeps the constant channel.
  void emitClassContract(const parser::Node &classDef,
                         llvm::StringRef symbolName = {},
                         const EmitOptions::SourceModule *source = nullptr);
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
  // Set while `collectClassFields` walks a method other than `__init__`: those
  // may only REFINE a field the constructor declared, never add one.
  bool collectFieldsRefineOnly = false;
  // `contractName` is the class's CANONICAL name -- `lib.R` for an imported
  // class, where the ClassDef's own name is only `R`. The walk asks the type
  // system about the class it is describing, and asking by the source spelling
  // answered nothing for every imported one.
  void collectClassFields(
      const parser::Node &classDef,
      llvm::SmallVectorImpl<std::string> &fieldNames,
      llvm::SmallVectorImpl<mlir::Type> &fieldTypes,
      bool includeAnnAssignDefaults = false, llvm::StringRef contractName = {},
      llvm::function_ref<void(llvm::ArrayRef<std::string>,
                              llvm::ArrayRef<mlir::Type>)>
          publishSoFar = nullptr,
      bool methodsDeclareFields = true);
  void collectStaticClassAssignments(
      const parser::Node &classDef, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values,
      llvm::SmallVectorImpl<mlir::Type> *types = nullptr);
  void collectStaticModuleAssignments(
      const parser::Node &moduleNode, llvm::SmallVectorImpl<std::string> &names,
      llvm::SmallVectorImpl<mlir::Attribute> &values) const;
  void collectModuleGlobals(const parser::Node &moduleNode);
  // The module's assignment targets nothing has bound yet, and the memoized
  // top-level signatures that read one once `collectModuleGlobals` bound it
  // (EmitterGlobals.cpp).
  llvm::StringSet<> unboundModuleTargets(const parser::Node &moduleNode) const;
  void forgetSignaturesReadingLateGlobals(const parser::Node &moduleNode,
                                          const llvm::StringSet<> &unbound);
  // Every name any function in `scope` declares `global`, at any depth.
  llvm::StringSet<> moduleGlobalDeclarations(const parser::Node &scope) const;
  // The element/key/value type an empty `[]`, `{}` or `set()` bound to `name`
  // is seeded with by the rest of the suite, or null when the seeds disagree
  // or there are none (EmitterStatements.cpp).
  // Binds every `NAME = <constant>` in the rest of the current suite into the
  // scope the caller just pushed; both forward scans need it.
  void preBindSuiteConstants();
  // The Callable an unannotated lambda bound to `name` takes from the calls
  // that follow it in this suite; null when they disagree or there are none.
  py::CallableType lambdaCallSeedContract(llvm::StringRef name,
                                          const parser::Node &lambda);
  // The union `NAME = None` takes when the rest of the suite binds NAME to
  // something else; null when nothing does or the bindings cannot be typed.
  mlir::Type noneSeedUnionType(llvm::StringRef name);
  bool functionBindsName(const parser::Node &function, llvm::StringRef name);
  mlir::Type
  emptyLiteralSeedAcross(llvm::StringRef name, llvm::StringRef literalKind,
                         llvm::ArrayRef<SuiteCursor> base,
                         llvm::ArrayRef<const parser::Node *> callables);
  mlir::Type emptyLiteralSeedType(llvm::StringRef name,
                                  llvm::StringRef literalKind);
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
  // `T = TypeVar("T")` + `Generic[T]` is the PEP 695 declaration one spelling
  // back. Rewritten into `type_params` before anything reads the tree, so the
  // machinery that already compiles `class Stack[T]` compiles both.
  void desugarClassicGenerics(const parser::Node &moduleNode);
  // `class P(Protocol)` becomes a plain class and a base of every class (in
  // any module of the program) that has its members; EmitterProtocols.cpp.
  void desugarProtocols();
  // `@contextlib.contextmanager` defs become a generator and a def that
  // wraps it (EmitterWith.cpp).
  void desugarContextManagers(const parser::Node &moduleNode);
  void desugarMethodDecorators(
      const parser::Node &moduleNode,
      const EmitOptions::SourceModule *source = nullptr);
  // A method the desugar split: its body under `hiddenName`, and the cell
  // `cellName` holding what its decorators returned, which the method of the
  // written name forwards to.
  struct DecoratedMethod {
    const parser::Node *classDef = nullptr;
    const parser::Node *hiddenDef = nullptr;
    std::string className;
    std::string hiddenName;
    std::string cellName;
    std::vector<parser::NodePtr> decorators;
    parser::SourceRange range;
    bool typed = false;
    // The module that declares the class, null for the main module; there
    // the cell is the module global `<module>.<cellName>` and the class the
    // contract `<module>.<className>`.
    const EmitOptions::SourceModule *source = nullptr;
    std::string globalName;
    std::string contractName;
  };
  std::vector<DecoratedMethod> decoratedMethods;
  parser::NodePtr decoratedMethodApplication(const DecoratedMethod &method,
                                             parser::NodePtr subject);
  void applyMethodDecorators(const parser::Node &classDef);
  void applyClassDecorators(const parser::Node &classDef);
  std::optional<Value>
  tryEmitTestProgramWithModuleFixtures(const parser::Node &expr);
  bool typeDecoratedMethod(DecoratedMethod &method, bool reportFailure);
  void bindDecoratedMethodCell(const DecoratedMethod &method, mlir::Type type);
  // Parsed trees a desugar spliced into a module, alive as long as it is.
  std::vector<parser::NodePtr> synthesizedHelperTrees;
  // The classes that were declared as protocols, by their own name.
  llvm::StringSet<> protocolClassNames;
  // "Class\0Protocol" (contract names) -> why the class does not satisfy it.
  llvm::StringMap<std::string> protocolMisses;
  // Protocol contract name -> why isinstance() against it cannot be answered
  // exactly (empty when it can).
  llvm::StringMap<std::string> protocolIsinstance;
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
  // ⭐ A GENERIC FUNCTION'S NON-LITERAL DEFAULT IS ONE VALUE FOR EVERY
  // SPECIALIZATION, as CPython evaluates it once at the def: a module cell
  // named by the generic, typed by the declared parameter where that is
  // ground and by the default's own type where the parameter names a type
  // parameter. Null when neither is ground. (EmitterFunctions.cpp)
  struct GenericDefaultCell {
    std::string name;
    mlir::Type type;
  };
  std::optional<GenericDefaultCell> genericDefaultCell(const parser::Node &def,
                                                       unsigned slot);
  // The default expression of parameter `slot` (positionals first, then the
  // keyword-only ones), or null.
  static const parser::Node *defaultExpressionForSlot(const parser::Node &def,
                                                      unsigned slot);
  // Why a call that leaves out a generic function's default cannot read the
  // shared cell at this specialization; empty when it can.
  std::string genericOmittedDefaultMismatch(const parser::Node &def,
                                            py::CallableType specialized,
                                            unsigned positionalGiven,
                                            llvm::ArrayRef<CallKeywordType>
                                                keywords);
  void emitDelete(const parser::Node &statement);
  // py.keep_alive for `value` when the program keeps locals alive and the
  // value can hold an object that observes its release.
  void emitKeepAlive(const parser::Node &at, Value value,
                     mlir::Type observedAs = {});
  // py.global.clear for a main-module global at the end of the program,
  // which is when CPython clears the module's names.
  void emitGlobalClear(const parser::Node &at, llvm::StringRef name);
  // py.keep_alive for every local of the frame, last bound first: CPython
  // clears a frame's locals in reverse slot order when it returns.
  void emitFrameExitKeepAlives(const parser::Node &at);
  // The frame's local names in CPython's slot order: parameters, then the
  // names it binds in the order they first appear.
  static std::vector<std::string> frameLocalOrder(const parser::Node &callable);
  std::vector<std::string> currentFrameLocals;
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
  // A function imported under the name (`from helpers import range`): a
  // competing callable that outranks a class of that spelling, as a
  // top-level `def` does.
  bool importBindsCallable(llvm::StringRef name) const;
  // A call to the builtin `name` that the program has not rebound: the gate
  // every builtin interception opens with.
  bool callsUnshadowedBuiltin(const parser::Node *calleeNode,
                              llvm::StringRef name) const;
  // The attribute name a `"..."` literal argument spells, when it is one.
  std::optional<llvm::StringRef>
  literalStringArgument(const parser::Node *node);
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
  bool iteratesBySequenceProtocol(mlir::Type type);
  void appendSequenceProtocolRead(parser::NodePtr source, parser::NodePtr index,
                                  const std::string &element,
                                  const std::string &done,
                                  std::vector<parser::NodePtr> &body,
                                  parser::SourceRange range);
  // `a, b = xs`: the length comparison CPython's UNPACK_SEQUENCE makes
  // (EmitterStatements.cpp, beside the assignment target walk it guards).
  void emitStarredUnpack(const parser::Node &target,
                         const std::vector<parser::NodePtr> &elements,
                         std::size_t starIndex, Value source);
  void emitUnpackArityCheck(const parser::Node &target, Value source,
                            std::size_t expected);
  bool tryEmitFileLineFor(const parser::Node &statement);
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
  // Rewrites a generator body's `for` over an indexable source into an
  // index loop, whose position rides a frame lane (EmitterLoops.cpp).
  bool emitGeneratorIndexedFor(const parser::Node &statement,
                               const parser::Node &iterNode);
  // `__len__` + `__getitem__` with no `__iter__` is CPython's fallback
  // iteration protocol, which is the same index loop (EmitterLoops.cpp).
  bool emitSequenceProtocolFor(const parser::Node &statement,
                               const parser::Node &iterNode);
  bool emitIndexedFor(const parser::Node &statement,
                      const parser::Node &iterNode);
  // A class with `__iter__` and `__next__` is the iterator protocol; the loop
  // is written as the try/except the protocol is (EmitterLoops.cpp).
  bool emitSourceIteratorFor(const parser::Node &statement,
                             const parser::Node &iterNode);
  // `__iter__` written as a generator: the loop iterates the CALL, whose value
  // is the generator object (EmitterLoops.cpp).
  bool emitGeneratorDunderIterFor(const parser::Node &statement,
                                  const parser::Node &iterNode);
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
  // sorted(key=, reverse=) / list.sort(key=, reverse=): the keys taken
  // into a list, then the native stable sort over them.
  std::optional<Value> tryEmitSortSugar(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<std::string>
  emitDsuSortStatements(const parser::Node &anchor, parser::NodePtr source,
                        const LazyCallable *key, bool reverse, bool inPlace,
                        unsigned serial,
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
    // Specializations whose bodies are being emitted right now: a body
    // reaching a NEW instantiation of its own function nests one more.
    unsigned emittingSpecializations = 0;
  };
  // What one call gives a function's `*args: P.args` and `**kwargs:
  // P.kwargs`: the positional arguments beyond its parameters, and the
  // keywords none of them takes, each at the type it arrives as.
  struct ParamSpecCallShape {
    llvm::SmallVector<mlir::Type, 4> positional;
    llvm::SmallVector<CallKeywordType, 4> keywords;
  };
  std::optional<std::pair<std::string, py::CallableType>>
  ensureGenericSpecialization(const parser::Node &anchor,
                              GenericFunctionInfo &generic,
                              py::CallableType target,
                              const ParamSpecCallShape *paramSpecShape = nullptr);
  llvm::SmallVector<unsigned, 2>
  concretizeParamSpecCallables(const parser::Node &def, FunctionSignature &sig,
                               unsigned declaredCount,
                               llvm::ArrayRef<mlir::Type> packPositional,
                               llvm::ArrayRef<CallKeywordType> packKeywords,
                               llvm::ArrayRef<bool> needsForwarder);
  // Whether a value is a builtin function or a class: what only a forwarder
  // makes a callable value of.
  bool needsCallableForwarder(Value value);
  // Per parameter of the generic call being specialized: whether its
  // argument needs a forwarder (emitGenericCall sets it).
  llvm::SmallVector<bool, 4> paramSpecForwarderArguments;
  ParamSpecCallShape paramSpecCallShape(const FunctionSignature &sig,
                                        const CallOperands &operands) const;
  // Binds the `*args` and `**kwargs` of a body specialized at one call's
  // shape to packs of the parameters the call's arguments arrive in, which a
  // call in the body forwards one by one; the returned packs are dropped from
  // the forwarding maps once the body is emitted.
  llvm::SmallVector<mlir::Value, 2>
  bindParamSpecPacks(const parser::Node &callable, const FunctionSignature &sig,
                     mlir::Block *entry);
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
  // One AST node per declared positional parameter: the supplied arguments,
  // then the literal defaults standing for the omitted trailing ones.
  bool specializationArgumentNodes(
      const parser::Node &expr, const GenericFunctionInfo &info,
      llvm::SmallVectorImpl<const parser::Node *> &out) const;
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
  // The imported module whose body is being emitted, or null in the main
  // module. Only a diagnostic reads it: a name that does not resolve THERE
  // has a different cause than one that does not resolve here.
  const parser::Node *activeSourceModuleNode = nullptr;
  // "unresolved name" is the wrong sentence when the imported module DOES
  // bind the name at its top level and the binding is one this compiler
  // cannot carry across the import. Returns the honest sentence, or empty.
  std::string importedModuleBindingReason(llvm::StringRef name) const;
  // The diagnostic for a name nothing binds: an import's own reason, a
  // builtin this compiler has no value for, or "unresolved name".
  std::string unresolvedNameMessage(llvm::StringRef name) const;
  // `__name__` or `__file__` of the module the code being emitted is written
  // in, as a str constant.
  Value emitModuleDunder(const parser::Node &anchor, llvm::StringRef name);
  // Generic lookup for a callee spelling: the local registration first, then
  // the canonical import binding (imported generics register under their
  // canonical "<module>.<name>" symbol).
  GenericFunctionInfo *lookupGenericFunction(llvm::StringRef name);
  // Source module that defines class `className` ("<module>.<Class>"
  // contract names), null for main-module and manifest classes.
  const EmitOptions::SourceModule *
  sourceModuleForClass(llvm::StringRef className) const;
  std::optional<Value> tryEmitCurriedGenericCall(const parser::Node &expr,
                                                 const parser::Node &factoryCall);
  Value emitExcInfo(const parser::Node &expr);
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
  // A call that hands over a starred sequence whose length only the run time
  // knows, or a `**mapping` whose keys only the run time knows (EmitterCalls.cpp,
  // emitDynamicUnpack). The callee's parameters, and the two spellings of its
  // name CPython's messages use.
  struct DynamicUnpackTarget {
    // The parameters a positional argument can fill, the receiver excluded.
    llvm::SmallVector<std::string, 8> positionalNames;
    unsigned positionalOnlyCount = 0;
    llvm::SmallVector<std::string, 4> kwOnlyNames;
    // `tuple[U]` and `dict[str, U]`, or null when there is none.
    mlir::Type varargType;
    mlir::Type kwargType;
    std::string module;
    std::string qualname;
  };
  struct DynamicUnpackCall {
    // How many positional values the call gives before the starred one, and
    // those of them beyond the parameters (what *args receives first).
    unsigned explicitPositional = 0;
    llvm::SmallVector<Value, 4> extras;
    std::optional<Value> tail;
    // In call order; the mapping is the one with the empty name.
    llvm::SmallVector<std::pair<std::string, Value>, 4> keywords;
  };
  struct DynamicUnpackResult {
    std::optional<Value> vararg;
    std::optional<Value> kwargs;
    // The keywords that bind named parameters, in call order.
    llvm::SmallVector<std::pair<std::string, Value>, 4> named;
  };
  std::optional<DynamicUnpackResult>
  emitDynamicUnpack(const parser::Node &anchor,
                    const DynamicUnpackTarget &target,
                    const DynamicUnpackCall &call);
  // `operands` rewritten for a callee the emitter knows the parameters of: its
  // *args is one tuple and its **kwargs one fresh dict.
  bool normalizeDynamicUnpacks(const parser::Node &anchor,
                               const DynamicUnpackTarget &target,
                               CallOperands &operands);
  std::optional<DynamicUnpackTarget> dynamicUnpackTargetFor(
      const parser::Node &anchor, Value callee);
  DynamicUnpackTarget dynamicUnpackTargetForMethod(const MethodBinding &method);
  // A manifest export is C behind its declared parameter, so an argument on a
  // lower numeric rung converts there (EmitterCalls.cpp). Null when nothing
  // needs widening; the returned node owns the rewritten arguments.
  parser::NodePtr
  widenNumericArgumentsForManifestCall(const parser::Node &expr,
                                       llvm::StringRef binding,
                                       py::CallableType declared);
  std::optional<Value> tryEmitDynamicClassName(const parser::Node &expr);
  // What the class line says about one attribute name, for `hasattr` and the
  // default form of `getattr`.
  enum class AttributePresence {
    Absent,
    Present,
    // A field the program may read before a store fills it: the INSTANCE
    // answers, through `attributeProbe`.
    FieldMaybeUnset,
    // Absent here, and a subclass may define it.
    Undecided,
  };
  AttributePresence attributePresence(mlir::Type subject,
                                      llvm::StringRef attribute,
                                      std::string *undecidedBy = nullptr);
  // `__ly_hasattr$<n>(recv: C) -> bool`, one per (class, field): reads the
  // field inside `try` and answers whether the read raised.
  std::string attributeProbe(llvm::StringRef cls, llvm::StringRef field,
                             parser::SourceRange range);
  llvm::StringMap<std::string> attributeProbes;
  // Lookups by a COMPUTED name (EmitterCalls.cpp): the program's classes
  // deriving from `cls`, whether `candidate` is one of them, and the names a
  // class declares or inherits from the program's classes.
  llvm::SmallVector<std::string, 8>
  programSubclassesOf(llvm::StringRef cls) const;
  bool programClassDerivesFrom(llvm::StringRef candidate,
                               llvm::StringRef ancestor) const;
  llvm::StringSet<> programAttributeNames(llvm::StringRef cls) const;
  // `getattr(recv, name)(args)` and `hasattr(recv, name)` with `name`
  // computed: a synthesized function comparing it with every name a value of
  // the receiver's class can answer.
  std::optional<Value>
  tryEmitComputedGetattrCall(const parser::Node &expr,
                             const parser::Node *calleeNode);
  std::optional<Value> emitComputedHasattr(const parser::Node &expr,
                                           const parser::NodePtr &receiver,
                                           const parser::NodePtr &name);
  // Their bodies, written once every class is declared.
  std::vector<parser::NodePtr>
  computedGetattrBody(parser::SourceRange range, llvm::StringRef cls,
                      llvm::ArrayRef<std::string> hierarchy,
                      llvm::ArrayRef<mlir::Type> argumentTypes);
  std::vector<parser::NodePtr>
  computedHasattrBody(parser::SourceRange range, llvm::StringRef cls,
                      llvm::ArrayRef<std::string> hierarchy);
  // The receiver class a computed-name lookup can enumerate, with the
  // classes it may be; a diagnostic and nullopt when it cannot.
  std::optional<llvm::SmallVector<std::string, 8>>
  computedLookupHierarchy(const parser::Node &expr, llvm::StringRef what,
                          mlir::Type receiverType);
  llvm::StringMap<std::string> computedLookupDispatchers;
  // `dir(C)` / `dir(t)` / `dir(o)` (EmitterCalls.cpp): the names a plain
  // program class lists, nullopt with `why` for a class whose attributes are
  // not all in class bodies.
  std::optional<std::set<std::string>> classDirNames(llvm::StringRef cls,
                                                     std::string &why) const;
  std::optional<Value> tryEmitDirCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  Value emitPercentFormatArguments(const parser::Node &expr, Value operand,
                                   unsigned wanted);
  // `object.__repr__(x)` and its siblings: object's own slot, not x's.
  std::optional<Value> tryEmitObjectSlotCall(const parser::Node &expr,
                                             const parser::Node *calleeNode);
  // `BaseException.__str__(e)` and `__repr__`: the base's text, not e's.
  std::optional<Value> tryEmitBaseExceptionText(const parser::Node &expr,
                                                mlir::Type owner,
                                                llvm::StringRef methodName,
                                                const parser::NodePtr &subject);
  std::optional<Value> tryEmitHasattrCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::optional<std::string> unhashableClassName(mlir::Type type) const;
  bool refuseUnhashableKey(const parser::Node &site, mlir::Type type,
                           llvm::StringRef role);
  std::optional<Value> tryEmitNamedTupleReplace(const parser::Node &expr,
                                                const parser::Node *calleeNode);
  std::optional<Value> tryEmitSetattrCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::optional<Value> tryEmitGetattrCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::optional<Value> tryEmitCallableCall(const parser::Node &expr,
                                           const parser::Node *calleeNode);
  // Whether a class can only be itself: nothing in the program derives from
  // it, and it is not a builtin something derives from (bool is an int, an
  // exception with subclasses, object).
  bool classIsClosed(llvm::StringRef contract) const;
  // Whether the class a `type[C]` value names is decided by its static type:
  // C is closed, or the expression spells the class itself. Otherwise the
  // value names C or any subclass, and only the run time knows which.
  bool classValueIsExact(const parser::Node *expr, mlir::Type type) const;
  std::optional<Value> emitClassValueIdentity(const parser::Node &expr,
                                              Value lhs, Value rhs,
                                              bool negated);
  // The classes a `type[C]` value may name: every class of the program and
  // every builtin exception deriving from C, then C itself last.
  llvm::SmallVector<std::string, 8>
  classValueCandidates(llvm::StringRef contract) const;
  // The class value the next construction of its bound takes its class word
  // from: a builtin exception constructed through `type[Exception]` is built
  // by the bound's initializer with the class the value names.
  std::optional<Value> pendingConstructionClass;
  // One arm per candidate, chosen by identity with the class value; each arm's
  // value is coerced to `resultType`.
  // `capturedNames` are the caller's bindings the arms read.
  // `armMismatch`, when given, is asked for each class as its arm is written
  // -- which may be after every class is declared -- and a non-empty answer
  // makes the arm raise that TypeError instead.
  std::optional<Value> emitClassValueDispatch(
      const parser::Node &anchor, Value classValue, py::ContractType bound,
      mlir::Type resultType,
      llvm::function_ref<parser::NodePtr(llvm::StringRef spelling)> armFor,
      llvm::ArrayRef<std::string> capturedNames = {},
      std::function<std::string(llvm::StringRef spelling)> armMismatch = {});
  // A call to a function of its own that raises `exception(message)`, typed
  // as `resultType`: a raise where an expression is wanted.
  parser::NodePtr raisingCall(parser::SourceRange range,
                              llvm::StringRef exception,
                              const std::string &message,
                              mlir::Type resultType);
  // `t(args)` and `t.m(args)` for a class value the static type does not
  // decide.
  std::optional<Value> emitClassValueCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  // `t.attr` -- a class attribute -- for such a value, already emitted.
  std::optional<Value> emitClassValueAttribute(const parser::Node &expr,
                                               Value classValue,
                                               llvm::StringRef attribute);
  Value emitClassOfValue(const parser::Node &anchor, Value value,
                         mlir::Type resultType);
  // A `def` with PEP 695 type parameters of its own.
  static bool hasTypeParameters(const parser::Node &def);
  // A def whose body spells its own `*args` or `**kwargs` into a call
  // (`f(*args, **kwargs)`): their arity and names are the call site's.
  static bool forwardsItsPacks(const parser::Node &def);
  // Whether an argument of type `actual` may bind a parameter declared
  // `expected` at an inlined method's call (EmitterClasses.cpp).
  bool inlineArgumentFits(mlir::Type actual, mlir::Type expected);
  void markExcInfoCallsInsideHandlers(const parser::Node &node,
                                      bool insideHandler);
  // The call `emitExprExpected` is emitting under a ground expectation, and
  // that expectation: what a generic call's result is wanted as, which solves
  // a type parameter only the result names (emitGenericCall).
  const parser::Node *expectedResultCall = nullptr;
  mlir::Type expectedResultType;
  // `obj.attr` where a field shadows a class attribute: the field when it is
  // set, `type(obj).attr` otherwise (EmitterExpressions.cpp).
  Value emitFieldOrClassAttribute(const parser::Node &expr, Value object,
                                  llvm::StringRef attr, mlir::Type fieldType);
  // The defaults a call leaves out, as keyword arguments of their types, for
  // the parameters whose declared type still names a type parameter
  // (EmitterCalls.cpp).
  llvm::SmallVector<CallKeywordType, 2>
  omittedDefaultArgumentTypes(const parser::Node &def,
                              const FunctionSignature &sig,
                              unsigned positionalGiven,
                              llvm::ArrayRef<CallKeywordType> keywords);
  // The defs whose body is typed only where a call is: a generic one, and
  // one that forwards its packs. Neither has a standalone body.
  static bool specializedAtEachCall(const parser::Node &def) {
    return hasTypeParameters(def) || forwardsItsPacks(def);
  }
  // An isinstance/issubclass operand, or an element of a tuple of them, that
  // is a class value the static type does not decide.
  bool targetHasDynamicClassValue(const parser::Node *node);
  std::optional<Value>
  emitClassValueSubclassTest(const parser::Node &expr,
                             const parser::Node *subjectNode,
                             const parser::Node *targetNode,
                             bool subjectIsInstance);
  std::optional<Value> tryEmitTypeCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
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
                         const parser::Node *containerNode, Value container,
                         const parser::Node &sliceNode,
                         llvm::StringRef methodName,
                         std::optional<Value> payload);
  // The slice protocol's mutator (`__setslice__` / `__delslice__`) called
  // with `bounds`: the four parts of a written `a:b:c`, or one slice object.
  void emitSliceProtocolMutation(const parser::Node &target,
                                 const parser::Node *containerNode,
                                 Value container, llvm::ArrayRef<Value> bounds,
                                 llvm::StringRef methodName,
                                 std::optional<Value> payload);
  // `slice(a, b, c)` for a written `a:b:c`, None for each absent part -- the
  // object CPython's BUILD_SLICE makes.
  Value emitSliceObject(const parser::Node &sliceNode);
  bool isSliceObjectType(mlir::Type type) const;
  // `sliceObject` is a slice and the receiver's slice protocol method
  // (`__getslice__` / `__setslice__` / `__delslice__`) takes it -- a list
  // slices with it where a dict looks it up as a key.
  bool receiverSlicesWith(Value container, llvm::StringRef methodName,
                          Value sliceObject,
                          std::optional<Value> payload = std::nullopt) const;
  // A source class that defines `dunder` receives a written slice as a slice
  // object.
  bool classTakesSliceObject(Value container, llvm::StringRef dunder);
  std::vector<parser::NodePtr> synthesizedSliceObjects;
  std::optional<Value> tryEmitReprCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitFormatCall(const parser::Node &expr,
                                         const parser::Node *calleeNode);
  // f-string machinery: JoinedStr folds stringified pieces with str.__add__;
  // FormattedValue applies !r/!s/!a and dispatches __format__ statically.
  Value emitJoinedStr(const parser::Node &expr);
  // ⭐ A GUARD IS SPENT AT THE READ, and CHECKED there. Shared by the two
  // storages that are RE-READ rather than held: a field and a CELL. The tag
  // the guard saw is evidence about the past, so the read tests it, raises
  // where it fails, and refines where it holds. `subject` names what the
  // sentence is about; `subjectKind` ("attribute", "captured local",
  // "module global") names it, and an attribute's failure is AttributeError.
  // Whether `name` is a cell or a module global, whose proofs are recorded
  // by bare name and spent at each read rather than by unwrapping a value.
  bool proofIsSpentAtRead(llvm::StringRef name) const;
  // Whether a region-bound name may get a UNION slot (cleared for an `if`,
  // whose arms join through block arguments instead).
  bool regionSlotsTakeUnions = true;
  mlir::Type storeProvedMember(py::UnionType declared, mlir::Type stored) const;
  std::string cellTypeSpelling(mlir::Type type) const;
  bool callAssignsModuleGlobal(const parser::Node &call, llvm::StringRef name,
                               unsigned depth, mlir::Type proved);
  Value emitCheckedNarrowedRead(const parser::Node &anchor, Value raw,
                                mlir::Type proved, llvm::StringRef subject,
                                llvm::StringRef subjectKind);
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
  // The same question for the !r / !a direction, where the ladder is shorter:
  // only __repr__ answers it.
  bool canConvertType(mlir::Type type, int64_t conversion);
  // ⭐ THE ONE PLACE A UNION'S TAG IS TURNED INTO A BRANCH CHAIN. Three
  // producers wrote this by hand -- the stringify chain, the operator arms and
  // the len path -- and the shape is not tidiness: the inactive members' lanes
  // hold zeroed placeholders, so the arms must be BRANCHES and not a select
  // over eagerly-computed values, and a producer that forgot that would call a
  // method on a header nobody wrote. `perMember` is handed the UNWRAPPED
  // member value and returns this arm's answer; every answer is coerced to
  // `resultType`, which the caller has already decided the members join at.
  Value emitUnionMemberDispatch(const parser::Node &anchor, Value unionValue,
                                py::UnionType unionType, mlir::Type resultType,
                                llvm::function_ref<Value(Value)> perMember);
  Value emitUnionStringify(const parser::Node &anchor, Value value,
                           py::UnionType unionType, unsigned index,
                           int64_t conversion = 's');
  // str.format with a compile-time template: fields are matched against the
  // call arguments during emission (R4: literal templates are checked here).
  std::optional<Value> tryEmitStrFormatCall(const parser::Node &expr,
                                            const parser::Node *calleeNode);
  // printf-style % formatting on a str left operand.
  Value emitPercentFormat(const parser::Node &expr);
  // One numeric `%` conversion (`d`, `x`, `f`, ...) of `argument`, with the
  // `__format__` spec it maps to: per member for a union, CPython's TypeError
  // for a builtin that is not a number.
  Value emitPercentNumber(const parser::Node &expr, Value argument,
                          char conversion, const std::string &spec);
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
  // `a OP b` where a's type has no such operator and b's class provides the
  // REFLECTED one. Returns nothing when the operator is not reflectable or
  // the right operand is not a source class that defines it.
  // `x in obj` where obj's class has no `__contains__` but is iterable:
  // CPython falls back to iteration, and so does this, by rewriting to
  // `any(<x> == <element> for <element> in obj)` before the operands are
  // emitted. Returns nothing when the shape does not apply.
  // The `:=` targets inside `node`, typed under `hints`, for the names that
  // are not bound in the enclosing scope yet. PEP 572 binds them THERE, and a
  // region that binds one needs the slot before it runs.
  llvm::StringMap<mlir::Type>
  walrusTargetsToBind(const parser::Node &node,
                      const llvm::StringMap<mlir::Type> &hints);
  std::optional<Value> tryEmitIterableMembership(const parser::Node &expr);
  // `x in s` for a str `s` and an `object` `x`: CPython's run-time check of
  // the left operand, through one helper per program.
  std::optional<Value> tryEmitObjectInStr(const parser::Node &expr);
  std::string objectInStrHelper;
  std::optional<Value> tryEmitReflectedBinary(const parser::Node &anchor,
                                              llvm::StringRef method,
                                              Value lhs, Value rhs,
                                              bool leftInferenceSucceeded);
  std::optional<Value> tryEmitClassDunder(const parser::Node &anchor,
                                          Value receiver,
                                          llvm::StringRef dunder,
                                          llvm::ArrayRef<Value> positional = {},
                                          bool *refused = nullptr);
  // A class's `__len__` as `len` and a truth test read it: through int's
  // length check.
  std::optional<Value> tryEmitClassLength(const parser::Node &anchor,
                                          Value receiver);
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
  // Adopts a declared container type for an argument that came back with
  // erased arguments -- the `set()` spelling (EmitterClasses.cpp).
  // A fresh display literal argument retyped to the solved generic formal.
  Value adoptSolvedDisplay(Value value, mlir::Type declared,
                           const parser::Node *node);
  Value adoptDeclaredContainer(Value value, mlir::Type declared,
                               const parser::Node &anchor);
  Value emitInlineMethodCall(const parser::Node &expr, Value receiver,
                             const MethodBinding &method);
  // A large method CALLED through a symbol instead of copied into the call
  // site; nullopt when the copy is what this call needs (EmitterClasses.cpp).
  // A generic method's solution at one call: its signature with the type
  // parameters substituted, and what each parameter was solved as.
  struct MethodSpecialization {
    const FunctionSignature &signature;
    const std::map<std::string, mlir::Type> &typeArguments;
  };
  std::optional<Value>
  tryCallLargeMethod(const parser::Node &anchor, Value receiver,
                     const MethodBinding &method,
                     llvm::ArrayRef<Value> positional,
                     const KeywordArguments &keywords,
                     const MethodSpecialization *specialization = nullptr);
  // The symbol of a generic method's body at one solution, `self` typed
  // `signature`'s first parameter, emitted on the first ask.
  std::string specializedMethodBody(const MethodBinding &method,
                                    const FunctionSignature &signature,
                                    const std::map<std::string, mlir::Type> &typeArguments,
                                    bool selfIsExact);
  // The symbol of `method`'s body with `self` typed `receiverType` and NOT
  // assumed to be exactly that class, emitted on the first ask.
  std::string openMethodBody(const MethodBinding &method,
                             mlir::Type receiverType,
                             const FunctionSignature &signature);
  Value emitInlineMethodBody(const parser::Node &anchor, Value receiver,
                             bool bindDescriptorReceiver,
                             const MethodBinding &method,
                             llvm::ArrayRef<Value> positional,
                             const KeywordArguments &keywords,
                             std::optional<Value> collectedVararg = std::nullopt,
                             std::optional<Value> collectedKwargs = std::nullopt);
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
  // `__getslice__` called with `bounds`: the four parts of a written `a:b:c`,
  // or one slice object.
  Value emitSliceProtocolRead(const parser::Node &expr, Value container,
                              llvm::ArrayRef<Value> bounds);
  Value emitAttribute(const parser::Node &expr);
  // A written `obj.name` where `name` is a manifest method only the emitter
  // calls (`__getslice__`, `__sort_by_keys__`, ...) on a builtin type: the
  // diagnostic CPython's AttributeError would be. True when refused.
  bool refuseInternalManifestMethod(const parser::Node &attribute);
  Value emitAwait(const parser::Node &expr);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable);
  // The element type an EMPTY container literal inside `literal` takes from
  // its siblings; null unless `element` is one (EmitterExpressions.cpp).
  mlir::Type siblingExpectationFor(const parser::Node &literal,
                                   const parser::Node *element,
                                   bool forKey);
  std::optional<Value> tryEmitUnpackedLiteral(const parser::Node &expr,
                                              mlir::Type expected);
  Value emitContainerLiteral(const parser::Node &expr,
                             mlir::Type expected = {});
  Value emitSetLiteral(const parser::Node &expr, mlir::Type expected = {});
  // The set displays CPython uses as a frozenset constant without building a
  // set from it (ConstantSetFold.h), over every module this emitter emits.
  const llvm::DenseSet<const parser::Node *> &frozensetSetDisplays();
  std::optional<llvm::DenseSet<const parser::Node *>> frozensetSetDisplaysMemo;
  // `and`/`or` over non-bool operands: CPython's operand-value result,
  // restricted to operand combinations whose join is statically representable
  // (R1). The all-bool fast path stays in emitExpr.
  Value emitBoolOpValue(
      const parser::Node &expr, bool isAnd,
      const std::vector<parser::NodePtr> &operands,
      llvm::ArrayRef<mlir::Type> operandTypes = {},
      llvm::function_ref<void(unsigned)> proveEarlierOperands = nullptr);
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
  std::optional<Value> emitStaticBoolConstant(const parser::Node &anchor,
                                              llvm::StringRef binding);
  std::optional<Value> emitStaticIntConstant(const parser::Node &anchor,
                                             llvm::StringRef binding);
  std::optional<Value> emitLiteralTypeConstant(const parser::Node &anchor,
                                               mlir::Type type);
  // `str.lower` used as a value: the forwarder synthesized for it, memoized by
  // "<contract>.<method>" so one program builds one.
  std::optional<Value> tryEmitBoundManifestMethodObject(
      const parser::Node &anchor, Value object, llvm::StringRef methodName);
  std::optional<Value> tryWrapBuiltinFunctionValue(Value value,
                                                   py::CallableType target,
                                                   const parser::Node &anchor);
  std::optional<Value> emitBuiltinForwarder(llvm::StringRef binding,
                                            py::CallableType target,
                                            const parser::Node &anchor);
  // A class named where a callable value is expected: the forwarder
  // `return C(a0, ...)`, typed by the callable, memoized with the builtin
  // ones under "<type>|<callable>".
  std::optional<Value> tryWrapClassValue(Value value, py::CallableType target,
                                         const parser::Node &anchor);
  std::optional<Value> tryEmitManifestMethodObject(const parser::Node &anchor,
                                                   Value object,
                                                   llvm::StringRef methodName);
  llvm::StringMap<std::pair<std::string, mlir::Type>> manifestMethodObjects;
  // Bound-method forwarders, keyed by receiver type and method.
  llvm::StringMap<std::string> boundManifestMethodObjects;
  // Builtin-function forwarders, keyed by builtin and the callable they are.
  llvm::StringMap<std::pair<std::string, mlir::Type>> builtinFunctionObjects;
  Value emitFunctionObject(const parser::Node &anchor,
                           llvm::StringRef symbolName, mlir::Type type,
                           llvm::ArrayRef<Capture> captures);
  // R6 nonlocal cells: a boxed local's storage is an instance of a
  // synthesized one-field class ("__ly_cell$N" with field "v").
  static bool isCellContract(mlir::Type type);
  // Binds `name`'s type for inference from the type of its STORAGE: a cell or
  // slot binds its content, as every read of the name produces.
  void bindStoredLocalType(llvm::StringRef name, mlir::Type storage);
  bool typeValueIsStatic(mlir::Type type) const;
  struct ForeignFieldWrite {
    mlir::Type root;                // the owner path's root name, typed
    std::vector<std::string> path;  // attributes from the root to the owner
    std::string field;
    mlir::Type type;
  };
  // Stores `collectClassFields` saw into another class's field.
  std::vector<ForeignFieldWrite> foreignFieldWrites;
  void applyForeignFieldWrites();
  bool zeroWidthSlotContent(mlir::Type content) const;
  mlir::Type cellContentType(mlir::Type cellType);
  // A cell that also records whether its content was ever written: the storage
  // for a name only SOME paths bind. Reading one is guarded.
  bool cellTracksBinding(mlir::Type cellType) const;
  // Bind, in the current scope, a maybe-unbound slot for every name these
  // bodies assign that the scope does not already have -- so the binding
  // survives the region and a read before it raises instead of failing to
  // resolve. Names whose type cannot be settled from the source are left
  // alone and keep the unresolved-name diagnostic.
  // `boundWithKnownType` names are bound with the type given rather than one
  // inferred from the bodies -- a loop target, whose type comes from the
  // iterable and not from any assignment inside the region.
  void bindConditionallyAssignedLocals(
      const parser::Node &anchor,
      llvm::ArrayRef<const std::vector<parser::NodePtr> *> bodies,
      const llvm::StringMap<mlir::Type> *inferenceHints = nullptr,
      const llvm::StringMap<mlir::Type> *boundWithKnownType = nullptr);
  bool nameIsReadAfterCurrentStatement(llvm::StringRef name) const;
  // Should this loop's target keep its binding after the loop, the way
  // CPython's does? Only when this statement is the one place in the suite
  // that binds the spelling, so one slot with one type is the whole story.
  bool loopTargetOutlivesLoop(llvm::StringRef name,
                              const parser::Node &statement) const;
  // Conservative: true whenever the walk cannot see the whole remaining scope.
  bool nameMayBeReadAfterCurrentStatement(llvm::StringRef name) const;
  mlir::Type inferConditionalLocalType(
      llvm::ArrayRef<const std::vector<parser::NodePtr> *> bodies,
      llvm::StringRef name);
  mlir::Type ensureCellClass(mlir::Type contentType,
                             const parser::Node &anchor,
                             bool tracksBinding = false);
  Value emitCellAlloc(const parser::Node &anchor, Value initial,
                      bool tracksBinding = false);
  void emitUnboundLocalGuard(const parser::Node &anchor, const Value &cell,
                             llvm::StringRef name);
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
  // Deepest numeric-tower disagreement between a declared cell type and the
  // type being stored into it, container element types included.
  bool numericRepresentationMismatch(mlir::Type declared, mlir::Type assigned,
                                     mlir::Type &declaredLeaf,
                                     mlir::Type &assignedLeaf) const;
  // A bool -- or a union of bool and int -- met where int is declared by a
  // storage or a callee that cannot take it as itself, and the int it is
  // there (EmitterExpressions.cpp).
  bool boolKeptAsInt(mlir::Type declared, mlir::Type assigned) const;
  Value rungsAsInt(Value value, const parser::Node &anchor);
  // The declared-cell name a container expression reads (a module global or a
  // class attribute), or empty when it is not one.
  std::string declaredCellNameFor(const parser::Node *container) const;
  // Diagnostic text when a call writes into a declared cell's ELEMENT storage
  // at a different numeric rung, or empty when it does not.
  // A constructor marked `ly.typing.keeps_arguments` given an argument of
  // another numeric rung than the parameter keeps it: the reason to refuse,
  // or empty.
  std::string keptArgumentRepresentationMismatch(
      llvm::StringRef className, mlir::Type instanceType,
      const CallInferenceResult &inference,
      llvm::ArrayRef<mlir::Type> argumentTypes) const;
  std::string cellElementRepresentationMismatch(
      const parser::Node *containerNode, mlir::Type containerType,
      const CallInferenceResult &inference,
      llvm::ArrayRef<mlir::Type> argumentTypes) const;
  mlir::Value emitBoolValue(Value value, const parser::Node &anchor);
  mlir::Value emitNumberTruth(Value value, const parser::Node &anchor);
  // Ints and floats an inlined method bound to a parameter declared
  // `object`: their truth test is an object's, which R1 does not refuse.
  llvm::DenseSet<mlir::Value> numbersHeldAsObject;
  // The values an inlined method packed into its `*args` and `**kwargs`,
  // keyed by the pack's own value: a call in the body that spells the pack
  // forwards these one by one, with the arity and the names the call site
  // gave.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<Value, 4>>
      staticPositionalPacks;
  llvm::DenseMap<mlir::Value,
                 llvm::SmallVector<std::pair<std::string, Value>, 4>>
      staticKeywordPacks;
  // Whether `def` reads its `**name` dict and nothing else: forwards it,
  // looks keys up, iterates it -- so the names the call gave are still its
  // names wherever the body forwards it.
  static bool keywordPackIsOnlyRead(const parser::Node &def,
                                    llvm::StringRef name);
  Value emitKeywordPackDict(
      const parser::Node &anchor,
      llvm::ArrayRef<std::pair<std::string, Value>> members,
      mlir::Type declared);
  // A copy of `lambda` whose calls spell out the static packs they forward,
  // through names bound to the packed values; null when it forwards none.
  parser::NodePtr lambdaWithPacksSpelledOut(const parser::Node &lambda);

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
  // The main module's, which `sourceName` is swapped away from while an
  // imported module's code is emitted.
  std::string mainSourceName;
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
  // The `d(f)` applications a decorated module def stands for, kept alive
  // because the module globals point at them.
  std::vector<parser::NodePtr> decoratorApplications;
  // While the decoration itself is being emitted, the subject name resolves to
  // the emitted SYMBOL rather than to the cell it is about to fill.
  std::string decoratorSubjectName;
  static bool isRecognizedNonBindingDecorator(llvm::StringRef leaf);
  parser::NodePtr bindingDecoratorCallee(const parser::NodePtr &decorator);
  // Module-level defs whose decorators rebind them (`f = d(f)`), and those
  // where the rebinding was asked for and the application is no function.
  llvm::DenseSet<const parser::Node *> rebindingDecoratedDefs;
  llvm::DenseSet<const parser::Node *> attemptedDecoratedDefs;
  llvm::StringSet<> moduleClassNames;
  // Main-module top-level `def`s whose spelling is also a manifest builtin's
  // binding name (`def len`), mapped to the symbol they are emitted under.
  // py.binding_ref carries a NAME, and the runtime lowering resolves that name
  // against the manifest before it looks for a user func.func, so emitting the
  // user function as @len would hand every reference to the builtin. Renaming
  // the emitted symbol keeps the two entities distinguishable at the only
  // layer that can tell them apart -- the one that knows which is which.
  llvm::StringMap<std::string> shadowedBuiltinSymbols;
  // What CPython calls each `def`, `lambda` and `class` of the program's
  // modules: its module, `__qualname__` and `__name__`, read off the tree once
  // the desugars are done (recordPythonDefinitionNames). A desugar that
  // renames a def it split out says what the def is called in
  // `pythonNameOverrides`.
  struct PythonDefinitionName {
    std::string module;
    std::string qualname;
    std::string name;
  };
  llvm::DenseMap<const parser::Node *, PythonDefinitionName>
      pythonDefinitionNames;
  llvm::DenseMap<const parser::Node *, std::string> pythonNameOverrides;
  // A class written in a function, declared at its module's top level
  // (EmitterLocalClasses.cpp): the qualified-name prefix of where it was
  // written, and the ones refused there.
  llvm::DenseMap<const parser::Node *, std::string> hoistedClassPrefixes;
  llvm::DenseSet<const parser::Node *> refusedLocalClasses;
  void hoistFunctionLocalClasses(const parser::Node &moduleNode,
                                 llvm::StringRef moduleName);
  void recordPythonDefinitionNames(const parser::Node &moduleNode,
                                   llvm::StringRef moduleName);
  // Each function emitted for a definition, under the symbol the emitter
  // names it by, in emission order -- the order its `.N` is given in.
  struct EmittedPythonDefinition {
    std::string symbol;
    std::string module;
    std::string qualname;
  };
  std::vector<EmittedPythonDefinition> emittedPythonDefinitions;
  void manglePythonDefinitionSymbols();
  // The call whose traceback frame must draw no `~~~^^^` underline, recognized
  // while its statement is emitted and read back by `loc`.
  //
  // ⛔ TRAVELS IN THE LOCATION, NOT IN AN ATTRIBUTE ON THE CALL. The frame is
  // pushed by installPythonExceptionCleanupFrames, which runs on LLVM IR after
  // every py op is gone; a location survives that far and an op attribute does
  // not. It is decided HERE rather than re-derived from the source text at
  // runtime because CPython decides it from the statement's AST -- `return
  // f(x)` and `y = f(x)` show no anchors, `return [f(x)][0]` does -- and the
  // AST exists only in this frame.
  const parser::Node *anchorlessCall = nullptr;
  // One entry per method body the emitter is currently writing INTO its caller.
  // A traceback frame comes from an LLVM function, and an inlined body has none
  // of its own, so the frames it would have contributed are recorded here and
  // ride out in the location (`ly.source.function`, `ly.source.inline_at`).
  // Without them `b.bad()` printed ONE frame where CPython prints two, and gave
  // the surviving one the CALLER's name against the CALLEE's line.
  struct InlineFrame {
    // The method being inlined, as a traceback names it: CPython shows
    // `co_name`, so `bad` and not `Box.bad`.
    std::string calleeName;
    // Where the call that brought us here is written, and in which function.
    // An empty `callerName` means the enclosing LLVM function, which only the
    // lowering can name.
    std::string callerName;
    // The file that call is written in. A method of another module is
    // emitted under ITS module's `sourceName`, so the caller's has to be
    // remembered here or the caller's frame prints the callee's file.
    std::string callerFile;
    std::int32_t line = 0;
    std::int32_t column = 0;
    std::int32_t endLine = 0;
    std::int32_t endColumn = 0;
    bool noAnchor = false;
  };
  llvm::SmallVector<InlineFrame, 4> inlineFrames;
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
  // Nonzero while a module's names are being bound for an emission in its
  // scope (`emitInDefiningModuleScope`).
  unsigned definingScopeSetupDepth = 0;
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
  llvm::StringSet<> frozenDataclassContracts;
  const std::string *frozenInitContract = nullptr;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classStaticAttrBindings;
  llvm::StringMap<llvm::StringMap<MethodBinding>> classMethodBindings;
  // ⭐ A METHOD BODY EMITTED AFTER EVERY CLASS IS DECLARED. Two sibling
  // subclasses that both call a base-typed method could not be compiled in any
  // order: the first one's body needs the second's method bindings, which its
  // own `emitClassContract` has not filled yet, and swapping them only swaps
  // which of the two is refused. The bindings are all that a body needs from
  // another class, and they are registered before the bodies are, so the top
  // level declares every class first and drains this queue afterwards.
  struct DeferredMethodBody {
    const parser::Node *statement = nullptr;
    FunctionSignature signature;
    std::string symbolName;
    std::string kind;
    std::string contractName;
    // The module the class was declared in, when it is not the one being
    // emitted: its body reads that module's names.
    const EmitOptions::SourceModule *source = nullptr;
    // The class whose type parameters the body spells.
    std::string definer;
  };
  std::vector<DeferredMethodBody> deferredMethodBodies;
  // The module `emitInDefiningModuleScope` is emitting in, if any.
  const EmitOptions::SourceModule *activeSourceModule = nullptr;
  bool deferClassMethodBodies = false;
  void emitDeferredMethodBodies();
  // Canonical (resolved) base contract names per class, in declaration order.
  llvm::StringMap<llvm::SmallVector<std::string, 4>> classBaseNames;
  // C3 linearization per class (self first, canonical contract names).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classMros;
  // The module's class hierarchy as WRITTEN, filled by collectTopLevelBindings
  // before any emission, so that "does a subclass override this" does not
  // depend on where in the file it is asked.
  llvm::StringMap<llvm::SmallVector<std::string, 4>> declaredClassBases;
  // When each program class is created: an imported module's before the main
  // module's, each in the order its module writes them.
  llvm::StringMap<unsigned> declaredClassRank;
  unsigned importedClassRanks = 0;
  unsigned mainModuleClassRanks = 0;
  // The program classes that name `cls` among their bases, in creation order.
  llvm::SmallVector<std::string, 8> directSubclassesOf(llvm::StringRef cls);
  std::optional<Value> tryEmitSubclassesCall(const parser::Node &expr);

  // ---- `__doc__` (EmitterDocstrings.cpp)
  // Each definition's docstring as CPython 3.14 keeps it, recorded before any
  // desugar rewrites the trees: a def, class or lambda maps to its docstring,
  // or to nullopt when it has none. A definition missing from the map was
  // written by this compiler, and what CPython would answer for it is not
  // known here.
  llvm::DenseMap<const parser::Node *, std::optional<std::string>>
      definitionDocstrings;
  llvm::StringMap<std::optional<std::string>> moduleDocstrings;
  void recordDocstrings(const parser::Node &moduleNode,
                        llvm::StringRef moduleName);
  // What a `__doc__` read answers for one class, method or function.
  struct DocAnswer {
    enum class Kind { Text, None, Unknown, Cell };
    Kind kind = Kind::Unknown;
    // Text: the docstring. Unknown: why it is not known. Cell: the module
    // global holding what a decorated method's decorators returned.
    std::string text;
    bool operator==(const DocAnswer &other) const {
      return kind == other.kind && text == other.text;
    }
  };
  DocAnswer docOfDefinition(const parser::Node *definition,
                            llvm::StringRef shown) const;
  DocAnswer docOfClass(llvm::StringRef cls) const;
  DocAnswer docOfMethod(llvm::StringRef cls, llvm::StringRef method);
  // The answer every class of `cls`'s line gives for the method, when they
  // all give one.
  std::optional<DocAnswer> decidedMethodDoc(llvm::StringRef cls,
                                            llvm::StringRef method);
  bool docOfValueIsAnswered(mlir::Type type) const;
  parser::NodePtr docAnswerStatement(const DocAnswer &answer,
                                     parser::SourceRange range);
  std::optional<Value> tryEmitDocRead(const parser::Node &expr);
  Value emitDocConstant(const parser::Node &anchor, const DocAnswer &answer);
  std::string docDispatcher(llvm::StringRef key, llvm::StringRef prefix,
                            llvm::ArrayRef<std::pair<std::string, mlir::Type>>
                                params,
                            parser::SourceRange range,
                            std::function<std::vector<parser::NodePtr>()> body,
                            parser::NodePtr returns = nullptr);
  // `sys.exit()` / `sys.exit(None or str)` as `raise SystemExit(code)`.
  std::optional<Value> tryEmitSysExit(const parser::Node &call);
  // `e.code` of a SystemExit, computed from its `args`.
  std::optional<Value> tryEmitSystemExitCode(const parser::Node &expr);
  // `__name__` of a function value the static type does not name.
  std::string functionNameDispatcher(mlir::Type callable,
                                     parser::SourceRange range);
  std::optional<Value> tryEmitFunctionValueNameRead(const parser::Node &expr);
  std::string classDocDispatcher(llvm::StringRef cls,
                                 parser::SourceRange range);
  std::string methodDocDispatcher(llvm::StringRef cls, llvm::StringRef method,
                                  parser::SourceRange range);
  std::string functionDocDispatcher(mlir::Type callable,
                                    parser::SourceRange range);
  std::vector<parser::NodePtr>
  functionDocBody(mlir::Type callable, parser::SourceRange range);
  std::optional<Value> emitComputedDocRead(const parser::Node &expr,
                                           const parser::NodePtr &receiverNode,
                                           const parser::NodePtr &nameNode);
  std::vector<parser::NodePtr>
  computedDocBody(parser::SourceRange range, llvm::StringRef cls,
                  llvm::ArrayRef<std::string> hierarchy);
  // Bodies that walk every function the program has: emitted after the main
  // module's statements, the last code to be written.
  std::vector<std::function<void()>> lateDispatcherBodies;
  llvm::StringMap<llvm::StringSet<>> declaredClassMethods;
  llvm::StringMap<llvm::StringSet<>> declaredClassAttributes;
  // The names a class body makes readable WITHOUT an instance store: a plain
  // class-level assignment (`name = "R2"`), or a property. A base's FIELD of
  // that name is shadowed by them, where an annotation shadows nothing.
  llvm::StringMap<llvm::StringSet<>> declaredClassReadableNames;
  // The names a class's own methods store on `self`.
  llvm::StringMap<llvm::StringSet<>> declaredClassStoredFields;
  // Every attribute name any statement of the program stores to, through any
  // receiver -- computed on first use, after every module is parsed.
  const llvm::StringSet<> &programAttributeStoreNames() const;
  mutable llvm::StringSet<> programAttributeStores;
  mutable bool programAttributeStoresScanned = false;
  void recordClassBodyDeclarations(llvm::StringRef name,
                                   const parser::Node &classDef);
  // Every program class's ClassDef, by the name the declaration maps use.
  llvm::StringMap<const parser::Node *> declaredClassNodes;
  // The fields every construction of `cls` stores before `self` can escape
  // (EmitterClasses.cpp); nullopt when construction is not this walk's to
  // read (a dataclass, a NamedTuple), which means every field.
  std::optional<llvm::StringSet<>>
  fieldsStoredByConstruction(llvm::StringRef cls);
  llvm::StringMap<std::optional<llvm::StringSet<>>> constructionStoreMemo;
  // Whether a read of `field` through a value of static class `cls` -- an
  // instance of it or of any subclass -- always follows a store.
  bool fieldAlwaysStoredBeforeRead(llvm::StringRef cls, llvm::StringRef field);
  // Marks a field read the lowering must test for an empty slot.
  void markFieldReadMaybeUnset(mlir::Operation *read, mlir::Type objectType,
                               llvm::StringRef field);
  // How many except handler bodies enclose the statement being emitted. A bare
  // `raise` re-raises what a handler caught, so at zero there is nothing to
  // re-raise -- the question the lowering cannot ask, because `py.try`'s
  // regions are gone by then.
  unsigned exceptHandlerDepth = 0;
  // Fields declared by the class body itself (classFieldOrders holds the
  // MRO-merged instance layout).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classOwnFieldOrders;
  // The fields a method other than `__init__` creates (`self.data = ...` in a
  // test's setUp). They are instance attributes, not dataclass fields: CPython's
  // dataclass reads the class annotations.
  llvm::StringMap<llvm::StringSet<>> classFieldsDeclaredOutsideInit;
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
  // The suite `emitStatements` is walking and how far it has got, so an EMPTY
  // container literal can look FORWARD for the operations that seed it
  // (`emptyLiteralSeedType`). Nothing else may read these: they describe where
  // the walk is, not what it has decided.
  const std::vector<parser::NodePtr> *currentSuite = nullptr;
  std::size_t currentSuiteIndex = 0;
  // Every suite currently being emitted, innermost last, each with the index
  // just past the statement it is emitting. A forward look needs all of them:
  // "read after this loop" means the rest of this suite AND the rest of every
  // suite it sits inside, up to the scope it belongs to.
  llvm::SmallVector<std::pair<const std::vector<parser::NodePtr> *,
                              std::size_t>, 8>
      suiteStack;
  // Where the current CALLABLE's suites start in that stack. A name in a
  // nested function is a different binding, so the walk stops here rather
  // than reading the enclosing function's remainder.
  unsigned suiteStackFloor = 0;

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
      // ⛔ EXCEPT AN IMPORTED MODULE'S OWN GLOBALS, which are exactly what a
      // body being emitted here reads. Their keys are QUALIFIED (`lib.NAMES`)
      // and a main-module global's is bare, so they cannot shadow one -- and
      // hiding them left `return len(NAMES)` inside lib.py unresolved while
      // `lib.NAMES` from the importer resolved, which is one storage with two
      // answers.
      for (const auto &entry : globals)
        if (entry.getKey().contains('.'))
          emitter.moduleGlobals[entry.getKey()] = entry.getValue();
    }
    ImporterModuleScope(const ImporterModuleScope &) = delete;
    ImporterModuleScope &operator=(const ImporterModuleScope &) = delete;
    ~ImporterModuleScope() {
      // And what the scope DECLARED of them outlives it, for the same
      // reason: a decorated method's cell is typed in its module's scope
      // and read by every importer that inlines the method.
      for (const auto &entry : emitter->moduleGlobals)
        if (entry.getKey().contains('.') && !globals.count(entry.getKey()))
          globals[entry.getKey()] = entry.getValue();
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
  // Field paths a guard has proved non-None, and the payload type proved. A
  // read of one is CHECKED against the type at the read, never assumed: the
  // field is re-read at every use and a call in between may have replaced it,
  // and unwrapping a union whose tag has changed is a garbage pointer rather
  // than a wrong answer. The entries live only inside the guarded region.
  llvm::StringMap<mlir::Type> narrowedMemberTypes;
  bool suppressMemberNarrowing = false;
  void invalidateMemberNarrowings(const parser::Node &statement);
  // Can this call assign `<root>.<attr>`? Only what the callee's own body --
  // and, one hop on, a function it hands the same value to -- is SEEN to
  // assign counts. A callee this cannot resolve answers no, which leaves the
  // proof standing and the read CHECKED: the sound answer either way, and the
  // one that does not refuse a program because the compiler lost the callee.
  bool callAssignsMemberPath(const parser::Node &call, llvm::StringRef root,
                             llvm::StringRef attr, unsigned depth,
                             bool allowMethods, mlir::Type proved);
  bool defAssignsThroughParam(const parser::Node &def, llvm::StringRef param,
                              llvm::StringRef attr, unsigned depth,
                              mlir::Type proved);
  const parser::Node *moduleFunctionDef(llvm::StringRef name);
  llvm::StringMap<const parser::Node *> moduleFunctionDefs;
  bool moduleFunctionDefsBuilt = false;
  // ⭐ WHAT A STORE PROVES. `self.f = 5` on a `int | None` field proves the
  // field is an int on every path below it, which is the same kind of fact a
  // guard proves and is spent the same way -- at the READ, with a check. The
  // store cannot write it into `narrowedMemberTypes` directly because
  // `invalidateMemberNarrowings` runs after the statement and would erase it,
  // so it lands here and that walk installs it.
  llvm::StringMap<mlir::Type> memberNarrowingsFromStores;
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
  // Module-level names the module binds more than once, computed on first
  // use: a lambda that captures one of them would freeze a value the
  // program goes on to replace.
  llvm::StringSet<> reboundModuleNamesCache;
  bool reboundModuleNamesComputed = false;
  const llvm::StringSet<> &reboundModuleNames();
  // ⭐ `return NotImplemented` IS THE PROTOCOL'S FALLBACK, and this compiler
  // can take it statically. CPython's documented way to write a comparison
  // dunder is to hand back the singleton for an operand the method does not
  // handle; the interpreter then tries the reflected method and, if that
  // declines too, falls back -- to identity for `==`/`!=` and to TypeError for
  // the four orderings. The operand types are known here, so the fallback is
  // emitted directly instead of a tri-state riding a `-> bool` return.
  struct NotImplementedFallback {
    std::string method;
    std::string receiver;
    std::string other;
  };
  std::optional<NotImplementedFallback> notImplementedFallback;
  // The fallback statement `return NotImplemented` stands for, or null when
  // the enclosing function is not a comparison dunder.
  static std::optional<NotImplementedFallback>
  comparisonDunderFallback(llvm::StringRef methodName,
                           llvm::ArrayRef<std::string> positionalNames);
  parser::NodePtr notImplementedFallbackStatement(const parser::Node &statement);
  void applyFunctionDecorators(const parser::Node &statement);
  mlir::Type currentReturnType;
  mlir::Type currentGeneratorSendType;
  // The yield type the generator's ANNOTATION promises, when it has one. Each
  // `yield` is checked against it at its own site, where the flow facts a guard
  // proved are available and the whole-body walk's are not.
  mlir::Type currentGeneratorYieldType;
  // The bound-method wrapper emitted for (method body, receiver type, bound
  // class), reused by every later read of the same method off the same type.
  llvm::DenseMap<std::tuple<const parser::Node *, mlir::Type, mlir::Type>,
                 std::string>
      boundMethodWrappers;
  // The body being emitted is a coroutine's: `await` is legal in it.
  bool currentFunctionIsCoroutine = false;
  // The call an expression statement discards the value of: a host call's
  // result there is not converted (adaptJsHostResult).
  const parser::Node *discardedHostResult = nullptr;
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
  // The statements whose pending cells have been written: a cell registered
  // under one of them afterwards would never be.
  llvm::DenseSet<const parser::Node *> flushedDefaultCellOwners;
  // The module-level statement whose execution evaluates `def`'s defaults --
  // the def itself, or the class statement around a method -- in the module
  // `root`; null when there is none.
  static const parser::Node *defaultCellOwnerIn(const parser::Node &root,
                                                const parser::Node &def);
  // The name of a method's default cell. It depends on the method alone, so
  // a call site that reaches the method before its body is emitted names the
  // same cell.
  std::string methodDefaultCellName(const parser::Node &root,
                                    const parser::Node &owner,
                                    const parser::Node &def,
                                    unsigned slot) const;
  // The cell a copy of `method`'s body reads `slot`'s default from,
  // registering it under the class statement if no emission has yet.
  std::optional<std::string> methodDefaultCell(const MethodBinding &method,
                                               unsigned slot,
                                               mlir::Type declared);
  // The cell a METHOD's non-constant default was parked in, per (def node,
  // slot). An inlined method call reads this instead of re-emitting the
  // expression -- which is what made `def add(self, into: list[int] = [])`
  // build a fresh list per call. Module-level FUNCTIONS need no such map: their
  // call sites go through the callable's default-value attributes.
  llvm::DenseMap<const parser::Node *,
                 llvm::SmallVector<std::pair<unsigned, std::string>, 2>>
      methodDefaultCells;
  // Non-constant defaults of NESTED defs: evaluated once when the enclosing
  // execution reaches the def statement, carried as synthetic closure
  // captures (slot -> capture index) shared by every call in that execution.
  llvm::DenseMap<const parser::Node *,
                 llvm::SmallVector<std::pair<unsigned, unsigned>, 2>>
      nestedDefaultCaptures;
  // CPython evaluates a nested callable's non-constant defaults when the
  // statement (or the expression) that creates it runs, in the ENCLOSING
  // frame. Both spellings need that; see the note at the definition.
  void evaluateNestedDefaults(const parser::Node &function,
                              const FunctionSignature &signature,
                              llvm::SmallVectorImpl<Capture> &captures);
  unsigned syntheticFunctionCounter = 0;
  // The closed mutually-recursive nested-def group of the body being emitted
  // (`mutualNestedDefGroup`), keyed by source name: the symbol assigned to
  // each member BEFORE any of their bodies is emitted, so a member can reach
  // its siblings the way it already reaches itself -- by symbol, with no
  // captures and therefore no reference cycle. Empty for every other body.
  struct ClosedSibling {
    std::string symbol;
    const parser::Node *node = nullptr;
  };
  struct SiblingGroupState {
    llvm::StringMap<ClosedSibling> members;
    llvm::SmallVector<std::string, 4> captures;
  };
  SiblingGroupState closedSiblingGroup;
  std::string nestedFunctionSymbolName(llvm::StringRef name,
                                       const parser::Node &function);
  unsigned listCompCounter = 0;
  // Cell classes are synthesized once per (widened) content type.
  llvm::DenseMap<mlir::Type, mlir::Type> cellClassContracts;
  llvm::DenseMap<mlir::Type, mlir::Type> bindingCellClassContracts;
  llvm::StringSet<> bindingCellContractNames;
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
    // Whether `self` is the defining class itself: true of a standalone
    // method body, which every call inlines instead, but not of a decorated
    // method's body, which the decorator's result calls with an instance of
    // any subclass that inherits it.
    bool selfIsExact = true;
  };
  // The bodies desugarMethodDecorators hid behind their decorators.
  llvm::DenseSet<const parser::Node *> decoratedMethodBodies;
  // The defs written directly in a class body, recorded as the class is
  // declared.
  llvm::DenseSet<const parser::Node *> classBodyDefs;
  llvm::SmallVector<SuperContext, 4> superContexts;
  // Method bodies currently being inlined, innermost last. Inlining emits
  // BOTH arms of every branch, so a cycle in the inline graph has no base
  // case and expands forever: re-entering a body already on this stack is
  // always unbounded, never a recursion that would have terminated.
  llvm::SmallVector<const parser::Node *, 8> methodsBeingInlined;
  // What `openMethodBody` emitted, by method and receiver type.
  llvm::DenseMap<std::pair<const parser::Node *, mlir::Type>, std::string>
      openMethodBodies;
  // What `specializedMethodBody` emitted, by method and its specialized
  // callable, for a `self` taken as exact and for one that is not.
  llvm::DenseMap<std::pair<const parser::Node *, mlir::Type>, std::string>
      specializedMethodBodies[2];
  // AST nodes in a method's def: what a copy at its call site costs.
  llvm::DenseMap<const parser::Node *, unsigned> methodDefSizes;
  // The calls a computed-name dispatcher's arms make (`computedGetattrBody`),
  // which call a method of any size.
  llvm::DenseSet<const parser::Node *> dispatcherArmCalls;
};

} // namespace lython::emitter
