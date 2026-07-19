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
  mlir::Type boolProtocol() const;

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
                                 llvm::StringRef localName);
  bool bindSourceModuleName(llvm::StringRef module,
                            llvm::StringRef exportedName,
                            llvm::StringRef localName);
  bool bindSourceModuleReexport(const EmitOptions::SourceModule &source,
                                llvm::StringRef exportedName,
                                llvm::StringRef localName);
  bool bindSourceModuleStar(llvm::StringRef module,
                            const parser::Node &anchor,
                            bool diagnoseUnsupported);
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
  bool isModuleGlobalRead(llvm::StringRef name) const;
  bool isModuleGlobalWrite(llvm::StringRef name) const;

  void emitStatements(const std::vector<parser::NodePtr> *statements,
                      bool skipDeclarations = false);
  void emitStatement(const parser::Node &statement);
  void emitAssignTarget(const parser::Node &target, Value value);
  void emitIf(const parser::Node &statement);
  void emitMatch(const parser::Node &statement);
  void emitFor(const parser::Node &statement);
  void emitGeneratorExpFor(const parser::Node &statement,
                           const parser::Node &genexpr);
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
                          mlir::Block *headerBlock);
  llvm::SmallVector<mlir::Value, 4>
  loopCarriedBranchOperands(const parser::Node &anchor,
                            const LoopControlContext &loop, mlir::Block *target);
  void emitTry(const parser::Node &statement);
  void emitWith(const parser::Node &statement, bool async);
  void emitWithCleanup(const parser::Node &anchor, const WithCleanup &cleanup);
  void emitActiveCleanups(const parser::Node &anchor);

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
    llvm::DenseMap<mlir::Type, std::string> specializations;
  };
  std::optional<std::pair<std::string, py::CallableType>>
  ensureGenericSpecialization(const parser::Node &anchor,
                              GenericFunctionInfo &generic,
                              py::CallableType target);
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
  std::optional<Value> tryEmitIsInstanceCall(const parser::Node &expr,
                                             const parser::Node *calleeNode);
  std::optional<Value> tryEmitIntCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value> tryEmitFloatCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  Value emitFloatFromInt(const parser::Node &anchor, Value argument);
  std::optional<Value> tryEmitStrCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value> tryEmitListCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitPrintCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitReducerCall(const parser::Node &expr,
                                          const parser::Node *calleeNode);
  std::optional<Value> tryEmitLenCall(const parser::Node &expr,
                                      const parser::Node *calleeNode);
  std::optional<Value> tryEmitNextCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> tryEmitRoundCall(const parser::Node &expr,
                                        const parser::Node *calleeNode);
  std::optional<Value> tryEmitReprCall(const parser::Node &expr,
                                       const parser::Node *calleeNode);
  std::optional<Value> rejectStubSourceCall(const parser::Node &expr,
                                            llvm::StringRef symbol,
                                            bool instantiation);
  bool methodBindingBindsReceiver(const MethodBinding &method) const;
  Value emitDescriptorReceiver(const parser::Node &anchor, Value receiver,
                               const MethodBinding &method);
  Value emitMethodObject(const parser::Node &anchor, Value receiver,
                         const MethodBinding &method);
  // Operator protocol on a source-class receiver (x[i], x[i]=v, len(x),
  // i in x): the class method inlines with pre-built arguments, the same
  // dispatch as an explicit x.__getitem__(i) call.
  Value emitInlineOperatorCall(const parser::Node &anchor, Value receiver,
                               const MethodBinding &method,
                               llvm::ArrayRef<Value> positional);
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
  Value emitAttribute(const parser::Node &expr);
  Value emitAwait(const parser::Node &expr);
  Value emitAsyncioRunCall(const parser::Node &expr);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable);
  Value emitAwaitValue(const parser::Node &anchor, Value awaitable,
                       const AwaitInferenceResult &inference);
  Value emitContainerLiteral(const parser::Node &expr,
                             mlir::Type expected = {});
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
  parser::Diagnostics diagnostics;
  llvm::StringMap<Value> values;
  llvm::StringMap<PrimitiveConstant> primitiveConstants;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classFieldBindings;
  // Declaration order of each class's fields (classFieldBindings is
  // unordered); drives the synthesized field-record constructor.
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classFieldOrders;
  llvm::StringMap<llvm::StringMap<mlir::Type>> classStaticAttrBindings;
  llvm::StringMap<llvm::StringMap<MethodBinding>> classMethodBindings;
  // Canonical (resolved) base contract names per class, in declaration order.
  llvm::StringMap<llvm::SmallVector<std::string, 4>> classBaseNames;
  // C3 linearization per class (self first, canonical contract names).
  llvm::StringMap<llvm::SmallVector<std::string, 8>> classMros;
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
  // Names declared `global` in the function currently being emitted (writes
  // to them target the module global instead of a new local). Saved/restored
  // around each callable body.
  llvm::StringSet<> currentGlobalDecls;
  bool atModuleScope = false;
  mlir::Type currentReturnType;
  mlir::Type currentGeneratorSendType;
  std::string currentFunctionPrefix;
  std::vector<parser::NodePtr> synthesizedDefaultProviders;
  unsigned syntheticFunctionCounter = 0;
  unsigned listCompCounter = 0;
  llvm::SmallVector<WithCleanup, 8> activeWithCleanups;
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
};

} // namespace lython::emitter
