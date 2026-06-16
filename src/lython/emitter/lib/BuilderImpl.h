#pragma once

#include "Builder.h"
#include "Protocols.h"
#include "TypeInference.h"

#include "Ast.h"
#include "PyDialectTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace lython::emitter {

struct FunctionInfo;

struct Value {
  mlir::Value value;
  mlir::Type type;
  std::optional<std::string> exactClass = std::nullopt;
  std::optional<std::string> provenClass = std::nullopt;
  std::optional<mlir::Type> protocolConcreteType = std::nullopt;
  py::CallableType paramSpecArgs = {};
  py::CallableType paramSpecKwargs = {};
  std::shared_ptr<FunctionInfo> callableInfo = nullptr;
};

struct StaticRangeElements {
  std::vector<Value> values;
  mlir::Type elementType;
};

struct StaticRangeSpec {
  std::int64_t start = 0;
  std::int64_t stop = 0;
  std::int64_t step = 1;
  mlir::Type elementType;
};

struct ReturnedCallableSummary;

struct ClosureCapture {
  std::string name;
  Value value;
  mlir::Type storageType;
};

struct DictSubscriptTarget {
  Value container;
  Value key;
  mlir::Type valueType;
};

enum class TypeAliasParameterKind { TypeVar, TypeVarTuple, ParamSpec };

struct TypeAliasParameter {
  TypeAliasParameterKind kind = TypeAliasParameterKind::TypeVar;
  std::string name;
  mlir::Type bound;
  parser::NodePtr defaultValue;
};

struct TypeAliasInfo {
  std::vector<TypeAliasParameter> parameters;
  parser::NodePtr value;
};

struct FunctionInfo {
  std::string name;
  std::string symbolName;
  const parser::Node *definition = nullptr;
  std::vector<TypeAliasParameter> typeParameters;
  std::map<std::string, mlir::Type> typeSubstitutions;
  std::vector<std::string> argNames;
  std::vector<parser::NodePtr> defaultValues;
  std::size_t positionalOnlyCount = 0;
  std::size_t positionalCount = 0;
  std::vector<std::string> kwonlyNames;
  std::vector<parser::NodePtr> kwonlyDefaultValues;
  std::optional<std::string> varargName;
  mlir::Type varargType;
  py::CallableType varargParameterPack = {};
  std::optional<std::string> kwargName;
  mlir::Type kwargType;
  py::CallableType kwargParameterPack = {};
  llvm::SmallVector<mlir::Type> argTypes;
  std::vector<ClosureCapture> closureCaptures;
  mlir::Type resultType;
  py::CallableType signatureType;
  mlir::Type functionType;
  mlir::FunctionType nativeFunctionType;
  mlir::FunctionType asyncFunctionType;
  bool isNative = false;
  bool isAsync = false;
  bool isInitMethod = false;
  std::string methodKind;
  bool mutatesSelf = false;
  bool mayThrow = false;
  bool isGenericTemplate = false;
  bool isSpecialization = false;
  bool requiresCallableValue = false;
  std::optional<std::string> returnedExactClass;
  std::optional<std::size_t> returnedValueArgIndex;
  std::optional<std::size_t> returnedClassArgIndex;
  std::optional<std::string> returnedCallableSymbolName;
  std::optional<std::size_t> returnedCallableArgIndex;
  std::shared_ptr<ReturnedCallableSummary> returnedCallable;
};

struct ReturnedCallableSummary {
  FunctionInfo info;
  std::vector<std::optional<std::size_t>> closureCaptureArgIndices;
};

struct SyncIteratorResolution {
  Value iterable;
  FunctionInfo iterMethod;
  FunctionInfo nextMethod;
  mlir::Type iteratorType;
  mlir::Type elementType;
  bool iterReturnsReceiver = false;
};

struct FunctionSpecialization {
  FunctionInfo info;
  std::map<std::string, FunctionInfo> callableAliases;
};

struct CallArgumentTuples {
  Value posargs;
  Value kwnames;
  Value kwvalues;
};

struct StaticKeywordArg {
  const parser::Node *anchor = nullptr;
  std::string name;
  const parser::Node *value = nullptr;
  std::optional<Value> generatedValue = std::nullopt;
};

struct ClassInfo {
  std::string name;
  const parser::Node *definition = nullptr;
  std::string templateName;
  std::vector<TypeAliasParameter> typeParameters;
  std::map<std::string, mlir::Type> typeSubstitutions;
  std::vector<std::string> baseNames;
  std::vector<std::string> mro;
  std::map<std::string, mlir::Type> fields;
  std::map<std::string, std::string> methods;
  std::map<std::string, parser::NodePtr> methodNodes;
  std::map<std::string, parser::NodePtr> ownMethodNodes;
  bool isGenericTemplate = false;
  bool isGenericSpecialization = false;
  bool inheritanceResolved = false;
};

struct PrimitiveConstant {
  mlir::Type type;
  std::int64_t integerValue = 0;
  double floatValue = 0.0;
};

struct FunctionTypeComment {
  llvm::SmallVector<mlir::Type> argTypes;
  mlir::Type resultType;
};

struct LoopTarget {
  mlir::Block *breakBlock = nullptr;
  mlir::Block *continueBlock = nullptr;
  std::vector<std::string> breakNames;
  std::vector<std::string> continueNames;
};

const parser::Field *field(const parser::Node &node, std::string_view name);
const std::string *stringField(const parser::Node &node, std::string_view name);
std::optional<std::string> symbolField(const parser::Node &node,
                                       std::string_view name);
std::optional<std::vector<std::string>>
symbolListField(const parser::Node &node, std::string_view name);
const parser::NodePtr *nodeField(const parser::Node &node,
                                 std::string_view name);
const std::vector<parser::NodePtr> *nodeListField(const parser::Node &node,
                                                  std::string_view name);
const parser::FieldValue *valueField(const parser::Node &node,
                                     std::string_view name);
bool hasNodeListEntries(const parser::Node &node, std::string_view name);
bool hasTypeParams(const parser::Node &node);
std::optional<int> singletonKey(const parser::Node &node);
bool referencesName(const parser::Node &node, llvm::StringRef name);
bool referencesName(const std::vector<parser::NodePtr> &statements,
                    llvm::StringRef name);
void collectAssignedNames(const parser::Node &stmt, std::set<std::string> &out);
std::string typeString(mlir::Type type);

class Builder::Impl {
public:
  Impl(mlir::MLIRContext &context, std::string moduleName);

  EmitResult emit(const parser::Node &module);

private:
  std::string moduleName;
  mlir::MLIRContext &context;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  parser::Diagnostics diagnostics;
  struct NameBindingSnapshot {
    std::optional<Value> symbol;
    std::optional<FunctionInfo> callableAlias;
    std::optional<PrimitiveConstant> primitiveConstant;
  };

  std::map<std::string, Value> symbols;
  std::map<std::string, FunctionInfo> functions;
  std::map<std::string, FunctionInfo> callableFunctionsBySymbol;
  std::map<std::string, FunctionInfo> localFunctions;
  std::map<std::string, FunctionInfo> callableAliases;
  std::map<const parser::Node *, FunctionInfo> lambdaCallableInfos;
  std::map<std::string, FunctionSpecialization> functionSpecializations;
  std::set<std::string> emittedFunctionSpecializations;
  std::map<std::string, ClassInfo> classes;
  std::map<std::string, TypeAliasParameter> globalTypeVariables;
  std::map<std::string, std::string> genericClassSpecializations;
  std::set<std::string> emittedClasses;
  std::map<std::string, mlir::Type> typeAliases;
  std::map<std::string, TypeAliasInfo> genericTypeAliases;
  std::map<std::string, PrimitiveConstant> primitiveConstants;
  std::set<std::string> activeGlobalNames;
  std::set<std::string> activeNonlocalNames;
  std::map<std::string, mlir::Type> activeTypeSubstitutions;
  std::map<std::string, std::string> staticModules;
  std::map<std::string, std::pair<std::string, std::string>>
      staticModuleSymbols;
  std::map<std::string, std::string> staticAnnotationAliases;
  llvm::DenseMap<mlir::Value, std::vector<Value>> finiteListElements;
  llvm::DenseMap<const parser::Node *, mlir::Type> stableTypeInferenceCache;
  llvm::DenseMap<const parser::Node *, bool> typeInferenceDependencyCache;
  llvm::DenseMap<const parser::Node *, mlir::Type> annotationTypeCache;
  llvm::StringMap<mlir::Type> stringAnnotationTypeCache;
  llvm::DenseMap<mlir::Type, llvm::DenseMap<mlir::Type, bool>>
      typeAssignableCache;
  std::set<std::string> activeTypeAssignable;
  std::set<std::string> activeProtocolConformance;
  std::vector<LoopTarget> loopStack;
  std::vector<const parser::Node *> functionStack;
  std::vector<std::string> functionSymbolStack;
  mlir::Type currentReturnType;
  bool blockTerminated = false;
  bool inNativeFunction = false;
  bool inAsyncFunction = false;
  bool inModuleMain = false;
  unsigned exceptionContextDepth = 0;
  unsigned nestedFunctionCounter = 0;
  unsigned functionSpecializationCounter = 0;

  NameBindingSnapshot snapshotNameBinding(llvm::StringRef name) const;
  void restoreNameBinding(llvm::StringRef name, NameBindingSnapshot snapshot);
  void bindTemporaryName(llvm::StringRef name, Value value);

  mlir::Location loc();
  mlir::Location loc(const parser::Node &node);
  py::IntType intType();
  py::FloatType floatType();
  py::BoolType boolType();
  py::StrType strType();
  py::NoneType noneType();
  py::ExceptionType exceptionType();
  py::ExceptionCellType exceptionCellType();
  py::ClassType classType(llvm::StringRef className);
  mlir::Type coroutineType(mlir::Type resultType);
  mlir::Type taskType(mlir::Type resultType);
  mlir::Type futureType(mlir::Type resultType);
  py::DictType dictType(mlir::Type keyType, mlir::Type valueType);
  py::ListType listType(mlir::Type elementType);
  mlir::Type i32Type();
  mlir::Type i1Type();

  EmitResult finish();
  void error(const parser::Node &node, std::string message);

  mlir::ArrayAttr stringArrayAttr(llvm::ArrayRef<std::string> values);
  mlir::ArrayAttr typeArrayAttr(llvm::ArrayRef<mlir::Type> values);
  std::optional<py::ProtocolType>
  protocolType(llvm::StringRef protocolName,
               llvm::ArrayRef<mlir::Type> suppliedArguments);
  enum class AnnotationUse { Value, TypeContract };
  std::optional<mlir::Type> typeFromAnnotation(const parser::NodePtr &node);
  std::optional<mlir::Type>
  typeFromAnnotation(const parser::NodePtr &node,
                     const std::map<std::string, mlir::Type> &typeVariables);
  std::optional<mlir::Type>
  typeFromAnnotation(const parser::NodePtr &node,
                     const std::map<std::string, mlir::Type> &typeVariables,
                     std::set<std::string> &activeAliases);
  std::optional<mlir::Type>
  typeFromAnnotation(const parser::NodePtr &node,
                     const std::map<std::string, mlir::Type> &typeVariables,
                     std::set<std::string> &activeAliases,
                     AnnotationUse annotationUse);
  bool isTypeAliasMarker(const parser::NodePtr &node) const;
  bool isTypeVarDefinition(const parser::Node &stmt) const;
  bool isTypingName(const parser::Node &node, llvm::StringRef name) const;
  void scanTypeVariables(const parser::Node &moduleNode);
  std::optional<TypeAliasParameter>
  parseTypeParameter(const parser::Node &param, bool allowParamSpec);
  std::vector<TypeAliasParameter> parseTypeParameters(const parser::Node &owner,
                                                      bool allowParamSpec);
  std::vector<TypeAliasParameter>
  referencedFunctionTypeParameters(const parser::Node &function) const;
  std::vector<TypeAliasParameter>
  genericBaseTypeParameters(const parser::Node &classNode);
  bool bindTypeVariablesFromAnnotation(
      const parser::NodePtr &annotation, mlir::Type actual,
      const std::map<std::string, TypeAliasParameterKind> &parameterKinds,
      std::map<std::string, mlir::Type> &substitutions);
  std::optional<FunctionInfo> specializeGenericFunctionCall(
      const parser::Node &expr, const FunctionInfo &info,
      const std::vector<parser::NodePtr> &args,
      const std::map<std::string, mlir::Type> *explicitTypeArguments = nullptr);
  std::optional<FunctionInfo> specializeParamSpecForwardingFunctionCall(
      const parser::Node &expr, const FunctionInfo &info,
      const std::vector<parser::NodePtr> &args);
  std::string specializationTypeKey(mlir::Type type) const;
  std::optional<std::string>
  instantiateGenericClass(const parser::Node &anchor, llvm::StringRef name,
                          llvm::ArrayRef<mlir::Type> arguments);
  std::optional<std::string> instantiateGenericClassFromAnnotation(
      const parser::Node &anchor, llvm::StringRef name,
      llvm::ArrayRef<parser::NodePtr> arguments,
      const std::map<std::string, mlir::Type> &typeVariables,
      std::set<std::string> &activeAliases);
  bool typeAssignable(mlir::Type expected, mlir::Type actual);
  void refineProtocolFieldFromValue(
      ClassInfo &info, llvm::StringRef fieldName, const parser::Node &value,
      const std::map<std::string, mlir::Type> &knownNames = {});
  bool classSubtypeOf(llvm::StringRef derived, llvm::StringRef base) const;
  bool classLayoutCompatible(llvm::StringRef derived,
                             llvm::StringRef base) const;
  bool classConformsToProtocol(py::ClassType subtype,
                               py::ProtocolType protocol);
  bool classConformsToCallable(py::ClassType subtype,
                               py::CallableType callable);
  bool typeSubtypeOf(mlir::Type subtype, mlir::Type supertype);
  std::optional<std::string> mostSpecificKnownClass(const Value &value) const;
  std::optional<std::string> classFactForView(const Value &value) const;
  Value markExactClass(Value value, llvm::StringRef className) const;
  Value markProvenClass(Value value, llvm::StringRef className) const;
  std::optional<Value> concreteProtocolValue(const Value &value) const;
  Value viewClassAs(const parser::Node &anchor, Value value,
                    llvm::StringRef targetClass);
  Value applyReturnedClassSummary(Value value, const FunctionInfo &info) const;
  struct UnionMemberMatch {
    mlir::Type sourceMember;
    mlir::Type narrowedType;
  };
  std::vector<UnionMemberMatch>
  unionMembersMatchingType(py::UnionType unionType, mlir::Type tested,
                           bool requireLayoutCompatibleDowncast = true);
  // Inserts py.union.wrap when a member value flows into a union-typed
  // position. Returns the value unchanged for non-union expectations or
  // non-member actual types (the caller's typeAssignable check then reports
  // the mismatch).
  Value coerceToExpectedType(const parser::Node &anchor, Value value,
                             mlir::Type expected);
  // Recognizes `<name> is [not] None` and `isinstance(<name>, T)`.
  // For unions, matchType/complementType are the projected member views. For
  // class values, matchType is the proven subclass view and complementType is
  // usually absent because "not T" is not represented as a first-class type.
  struct NarrowingTest {
    std::string name;
    mlir::Type sourceType;
    mlir::Type matchType;
    mlir::Type complementType;
    bool negated = false;
    bool staticTruthKnown = false;
    bool staticTruth = false;
  };
  std::optional<NarrowingTest>
  matchUnionNarrowingTest(const parser::Node &test);
  // Resolves an isinstance(<expr>, <type>) call to (value node, member type).
  std::optional<std::pair<const parser::Node *, mlir::Type>>
  matchIsinstanceCall(const parser::Node &call);
  bool typeInferenceDependsOnEnvironment(const parser::Node &node);
  std::optional<typing::Term> typeTermFromType(mlir::Type type) const;
  std::optional<mlir::Type> typeFromTypeTerm(const typing::Term &term);
  std::optional<mlir::Type>
  inferExpressionTypeAlgorithmM(const parser::Node &expr);
  std::optional<mlir::Type> inferExpressionType(const parser::Node &expr);
  std::optional<mlir::Type>
  inferExpressionTypeUncached(const parser::Node &expr);
  mlir::Type typeFromClassAnnotation(llvm::StringRef className);
  mlir::Type listElementType(mlir::Type type);
  std::optional<std::pair<mlir::Type, mlir::Type>>
  dictKeyValueTypes(mlir::Type type);
  mlir::Type listLiteralElementTypeForExpected(mlir::Type expectedType);
  std::optional<std::pair<mlir::Type, mlir::Type>>
  protocolMappingKeyValueTypes(mlir::Type type);
  std::optional<std::pair<mlir::Type, mlir::Type>>
  dictLiteralKeyValueTypesForExpected(mlir::Type expectedType);
  bool dictStorageSupported(mlir::Type keyType, mlir::Type valueType);
  std::optional<std::string> classNameFromType(mlir::Type type);
  std::optional<std::string> staticSymbol(const parser::Node &node,
                                          llvm::StringRef moduleName) const;
  std::optional<std::string> primitiveTypeName(const parser::Node &node) const;
  std::optional<std::string> lyrtBuiltinName(const parser::Node &node) const;
  bool isBuiltinExceptionClass(llvm::StringRef name) const;
  bool isTensorConstructorCallee(const parser::Node &node) const;
  std::optional<std::int64_t> staticIndexValue(const parser::Node &node) const;
  std::optional<std::int64_t> staticPyIntValue(mlir::Value value) const;
  std::optional<unsigned> intWidthFromSubscript(const parser::Node &node);
  std::optional<PrimitiveConstant>
  primitiveIntConstructorConstant(const parser::Node &node);
  std::optional<PrimitiveConstant>
  primitiveScalarConstructorConstant(const parser::Node &node);
  bool isNativeDecorator(const parser::Node &node);
  py::CallableType functionSignatureType(
      llvm::ArrayRef<mlir::Type> argTypes, mlir::Type resultType,
      mlir::Type varargType = {}, llvm::ArrayRef<mlir::Type> kwonlyTypes = {},
      mlir::Type kwargType = {},
      llvm::ArrayRef<std::string> positionalNames = {},
      llvm::ArrayRef<char> positionalDefaults = {},
      llvm::ArrayRef<std::string> kwonlyNames = {},
      llvm::ArrayRef<char> kwonlyDefaults = {}, llvm::StringRef varargName = {},
      llvm::StringRef kwargName = {}, std::size_t positionalOnlyCount = 0);
  py::CallableType
  callableParameterPackFromSignature(py::CallableType signature);
  py::CallableType
  callableParameterPackFromTuple(llvm::ArrayRef<mlir::Type> positionalTypes);
  py::CallableType callableSignatureWithResult(py::CallableType pack,
                                               mlir::Type resultType);
  py::CallableType
  prependCallableParameterPack(llvm::ArrayRef<mlir::Type> prefix,
                               py::CallableType suffix);
  std::optional<py::CallableType>
  dropCallableParameterPrefix(py::CallableType pack, std::size_t prefixCount);
  std::optional<py::CallableType> callableParameterPackFromAnnotation(
      const parser::NodePtr &node,
      const std::map<std::string, mlir::Type> &typeVariables,
      std::set<std::string> &activeAliases);
  std::optional<py::CallableType> callableParameterPackFromAnnotation(
      const parser::NodePtr &node,
      const std::map<std::string, mlir::Type> &typeVariables,
      std::set<std::string> &activeAliases, AnnotationUse annotationUse);
  void refreshFunctionTypes(FunctionInfo &info);
  mlir::FunctionType asyncFunctionType(llvm::ArrayRef<mlir::Type> argTypes,
                                       mlir::Type resultType);
  mlir::Type awaitablePayloadType(mlir::Type type);
  bool lowerableAwaitableType(mlir::Type type) const;
  mlir::Type lowerableAwaitableValueType(const Value &value) const;
  mlir::Type methodAwaitableType(const FunctionInfo &method);
  Value awaitConcreteValue(const parser::Node &anchor, const Value &awaitable,
                           llvm::StringRef contextLabel);
  std::optional<mlir::Type> inferRangeTargetType(const parser::Node &iter);
  std::optional<mlir::Type> closureStorageType(mlir::Type type);
  Value materializeClosureStorage(const parser::Node &anchor,
                                  const ClosureCapture &capture);
  Value restoreClosureValue(const parser::Node &anchor, mlir::Value storage,
                            mlir::Type storageType, mlir::Type valueType);
  std::vector<ClosureCapture>
  collectNestedFunctionCaptures(const parser::Node &function);
  bool laterRebindsCapturedName(const parser::Node &function,
                                llvm::ArrayRef<ClosureCapture> captures);

  std::optional<FunctionInfo> parseFunctionInfo(
      const parser::Node &function,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr);
  std::optional<FunctionInfo> parseMethodInfo(
      const parser::Node &function, llvm::StringRef className,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr);
  std::optional<FunctionInfo> parseLambdaInfo(const parser::Node &lambda,
                                              mlir::Type expectedType);
  std::optional<mlir::Type> parameterType(
      const parser::Node &arg,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr,
      mlir::Type fallbackType = {});
  bool appendAnnotatedParameter(
      const parser::Node &arg, FunctionInfo &info, llvm::StringRef role,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr,
      mlir::Type fallbackType = {});
  bool collectCallableDefaults(
      const parser::Node &callable, const parser::Node &arguments,
      FunctionInfo &info,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr);
  std::optional<std::string> specializeClassForConstructorFields(
      const parser::Node &anchor, llvm::StringRef name,
      const FunctionInfo &init, llvm::ArrayRef<Value> userArgs);
  std::optional<std::string> specializeClassForConstructorFieldTypes(
      const parser::Node &anchor, llvm::StringRef name,
      const FunctionInfo &init, llvm::ArrayRef<mlir::Type> userArgTypes);
  std::optional<std::vector<std::string>>
  computeClassMro(llvm::StringRef className, const parser::Node &anchor,
                  std::set<std::string> &visiting);
  bool resolveClassInheritance(llvm::StringRef className,
                               const parser::Node &anchor,
                               std::set<std::string> &resolving);
  bool verifyInitFieldInitialization(const ClassInfo &info,
                                     const parser::Node &anchor);
  void scanClasses(const parser::Node &moduleNode);
  void scanTypeAliases(const parser::Node &moduleNode);
  void scanFunctions(const parser::Node &moduleNode);
  std::optional<FunctionTypeComment> parseFunctionTypeComment(
      const parser::Node &function,
      const std::map<std::string, mlir::Type> *typeVariables = nullptr);
  bool expressionMayThrow(const parser::Node &expr) const;
  bool statementMayThrow(const parser::Node &stmt) const;
  bool
  statementListMayThrow(const std::vector<parser::NodePtr> &statements) const;
  void propagateMayThrow(const parser::Node &moduleNode);
  py::ClassOp createClass(llvm::StringRef name, mlir::ArrayAttr baseNames = {});
  py::CallableFuncOp createFunc(llvm::StringRef name,
                                py::CallableType signature,
                                mlir::ArrayAttr argNames = {},
                                bool hasVararg = false, bool hasKwarg = false,
                                bool mayThrow = false,
                                mlir::ArrayAttr kwonlyNames = {},
                                mlir::ArrayAttr closureTypes = {});
  void addEntryBlock(py::CallableFuncOp func,
                     llvm::ArrayRef<mlir::Type> argTypes);
  void addEntryBlock(mlir::func::FuncOp func);
  void addAsyncEntryBlock(mlir::Operation *func,
                          llvm::ArrayRef<mlir::Type> argTypes);
  void emitPrelude();
  void scanStaticImports(const parser::Node &moduleNode);
  void scanReturnedCallableSummaries(const parser::Node &moduleNode);
  void scanMethodReturnedCallableSummaries();
  std::optional<std::string>
  directReturnedCallableSymbol(const parser::Node &function) const;
  std::optional<FunctionInfo>
  directReturnedNestedCallableMetadata(const FunctionInfo &outer,
                                       const parser::Node &function);
  std::optional<std::size_t>
  directReturnedCallableArgIndex(const FunctionInfo &info,
                                 const parser::Node &function) const;
  std::optional<std::size_t>
  directReturnedValueArgIndex(const FunctionInfo &info,
                              const parser::Node &function);
  std::optional<mlir::Type>
  directReturnedConcreteType(mlir::Type declaredResultType,
                             const parser::Node &function);
  std::optional<mlir::Type> directReturnedConcreteType(
      mlir::Type declaredResultType, const parser::Node &function,
      const std::map<std::string, mlir::Type> &knownNames);
  std::optional<std::string>
  directReturnedExactClass(const FunctionInfo &info,
                           const parser::Node &function);
  std::optional<std::size_t>
  directReturnedClassArgIndex(const FunctionInfo &info,
                              const parser::Node &function);
  std::shared_ptr<ReturnedCallableSummary>
  directReturnedNestedCallable(const FunctionInfo &outer,
                               const parser::Node &function);
  void emitMain(const parser::Node &moduleNode);
  void emitUserFunctions(const parser::Node &moduleNode);
  void emitFunctionDef(const parser::Node &function);
  void emitFunctionBody(const parser::Node &function, const FunctionInfo &info);
  void emitFunctionBody(
      const parser::Node &function, const FunctionInfo &info,
      const std::map<std::string, FunctionInfo> &initialCallableAliases);
  void emitNestedFunctionDef(const parser::Node &function);
  void emitAsyncFunctionDef(const parser::Node &function);
  void emitAsyncFunctionDef(
      const parser::Node &function, const FunctionInfo &info,
      const std::map<std::string, FunctionInfo> &initialCallableAliases);
  void emitClassDefs(const parser::Node &moduleNode);
  void emitClassDef(const parser::Node &classNode);
  void emitClassInfo(const parser::Node &classNode, const ClassInfo &info);
  void emitPendingGenericClassDefs();
  void emitMethodDef(const parser::Node &function, const FunctionInfo &info);
  void emitAsyncMethodDef(const parser::Node &function,
                          const FunctionInfo &info);

  void emitStatement(const parser::Node &stmt);
  void emitImport(const parser::Node &stmt, bool reportErrors = true);
  void emitImportFrom(const parser::Node &stmt, bool reportErrors = true);
  void emitDelete(const parser::Node &stmt);
  void emitAssign(const parser::Node &stmt);
  bool assignLiteralElementsToTarget(const parser::Node &stmt,
                                     const parser::Node &target,
                                     const parser::Node &source);
  void assignValueToTarget(const parser::Node &stmt, const parser::Node &target,
                           const Value &value,
                           const parser::Node *sourceNode = nullptr);
  void updateCallableAliasForBinding(llvm::StringRef name, const Value &value,
                                     const parser::Node *sourceNode);
  void assignAttributeValue(const parser::Node &stmt,
                            const parser::Node &target, const Value &value);
  std::optional<DictSubscriptTarget>
  emitDictSubscriptTarget(const parser::Node &target);
  void assignDictSubscriptValue(const parser::Node &stmt,
                                const DictSubscriptTarget &dictTarget,
                                const Value &value);
  void assignSubscriptValue(const parser::Node &stmt,
                            const parser::Node &target, const Value &value);
  void emitAttributeAssign(const parser::Node &stmt, const parser::Node &target,
                           const parser::Node &valueNode);
  void emitSubscriptAssign(const parser::Node &stmt, const parser::Node &target,
                           const parser::Node &valueNode);
  void emitAnnAssign(const parser::Node &stmt);
  void emitAugAssign(const parser::Node &stmt);
  void emitIf(const parser::Node &stmt);
  void emitFor(const parser::Node &stmt);
  void emitAsyncFor(const parser::Node &stmt);
  void emitForOverSequence(const parser::Node &stmt,
                           const std::string &targetName,
                           const parser::Node &iterable,
                           const std::vector<parser::NodePtr> &body,
                           const std::vector<parser::NodePtr> &orelse);
  void emitForOverIterator(const parser::Node &stmt,
                           const std::string &targetName,
                           const parser::Node &iterable, Value iterableValue,
                           const std::vector<parser::NodePtr> &body,
                           const std::vector<parser::NodePtr> &orelse);
  void emitWith(const parser::Node &stmt);
  void emitAsyncWith(const parser::Node &stmt);
  void emitWhile(const parser::Node &stmt);
  void emitMatch(const parser::Node &stmt);
  void emitBreak(const parser::Node &stmt);
  void emitContinue(const parser::Node &stmt);
  void emitTry(const parser::Node &stmt);
  bool emitStatementList(
      const std::vector<parser::NodePtr> &statements,
      const std::map<std::string, Value> &baseSymbols,
      const std::map<std::string, FunctionInfo> &baseCallableAliases);
  void emitReturn(const parser::Node &stmt);
  void emitRaise(const parser::Node &stmt);
  void emitAssert(const parser::Node &stmt);

  Value emitExpression(const parser::Node &expr);
  Value emitExpressionWithExpectedType(const parser::Node &expr,
                                       mlir::Type expectedType);
  Value emitLambda(const parser::Node &expr, mlir::Type expectedType);
  Value emitCondition(const parser::Node &expr);
  Value emitName(const parser::Node &expr);
  Value emitAwait(const parser::Node &expr);
  Value emitBoolOp(const parser::Node &expr);
  Value emitIfExp(const parser::Node &expr);
  Value emitNamedExpr(const parser::Node &expr);
  Value emitBinaryOperation(const parser::Node &expr, llvm::StringRef op,
                            const Value &lhs, const Value &rhs);
  Value emitBinOp(const parser::Node &expr);
  Value emitTupleLiteralMembership(const parser::Node &expr,
                                   const parser::Node &candidateNode,
                                   const parser::Node &tupleNode, bool negate);
  Value emitCompare(const parser::Node &expr);
  Value emitConstant(const parser::Node &expr);
  Value emitJoinedStr(const parser::Node &expr);
  Value emitFormattedValue(const parser::Node &expr);
  Value emitTemplateStr(const parser::Node &expr);
  Value emitInterpolation(const parser::Node &expr);
  Value emitUnaryOp(const parser::Node &expr);
  mlir::Value emitMatchEquality(const parser::Node &anchor,
                                const Value &subject, const Value &candidate);
  mlir::Value emitMatchPatternCondition(
      const parser::Node &pattern, const Value &subject,
      std::vector<std::pair<std::string, Value>> &captures);
  Value emitCall(const parser::Node &expr);
  Value emitAttribute(const parser::Node &expr);
  Value emitDict(const parser::Node &expr, mlir::Type preferredKeyType = {},
                 mlir::Type preferredValueType = {});
  Value emitDictComprehension(const parser::Node &expr,
                              mlir::Type preferredKeyType = {},
                              mlir::Type preferredValueType = {});
  Value emitList(const parser::Node &expr,
                 mlir::Type preferredElementType = {});
  Value emitListFromValues(const parser::Node &anchor,
                           llvm::ArrayRef<Value> values,
                           mlir::Type elementType = {});
  Value emitListComprehension(const parser::Node &expr,
                              mlir::Type preferredElementType = {});
  std::optional<std::vector<Value>>
  emitStaticGeneratorElements(const parser::Node &expr, llvm::StringRef context,
                              mlir::Type preferredElementType = {});
  std::optional<StaticRangeSpec>
  staticRangeSpec(const parser::Node &rangeCall, llvm::StringRef context,
                  mlir::Type preferredElementType = {});
  std::optional<std::size_t>
  staticRangeLength(const StaticRangeSpec &spec) const;
  std::optional<StaticRangeElements>
  emitStaticRangeElements(const parser::Node &rangeCall,
                          llvm::StringRef context,
                          mlir::Type preferredElementType = {});
  std::optional<std::vector<Value>>
  finiteSequenceElements(const parser::Node &anchor, const Value &sequence,
                         llvm::StringRef context);
  Value emitTupleLiteral(const parser::Node &expr,
                         mlir::Type expectedTupleType = {});
  Value emitSubscript(const parser::Node &expr);
  Value emitGetItemAccess(const parser::Node &expr, Value container,
                          const parser::Node &index,
                          llvm::StringRef contextLabel);
  Value emitTensorConstructor(const parser::Node &expr);
  Value emitTensorMatmul(const parser::Node &expr, const Value &lhs,
                         const Value &rhs);
  Value emitPrimitiveIntConstructor(const parser::Node &expr,
                                    PrimitiveConstant constant);
  Value
  emitPrimitiveScalarConstructor(const parser::Node &expr,
                                 mlir::Type targetType,
                                 const std::vector<parser::NodePtr> &args);
  std::optional<std::vector<const parser::Node *>> expandStaticCallArgs(
      const parser::Node &expr, const std::vector<parser::NodePtr> &args,
      std::optional<std::size_t> positionalLimit, llvm::StringRef calleeName);
  std::optional<std::vector<StaticKeywordArg>>
  expandStaticCallKeywords(const parser::Node &expr,
                           llvm::StringRef calleeName);
  std::optional<std::vector<Value>>
  emitStaticArguments(const parser::Node &expr, const FunctionInfo &info,
                      const std::vector<parser::NodePtr> &args,
                      std::size_t firstFormal = 0);
  bool hasCallableMetadata(const FunctionInfo &info) const;
  bool hasCallableFormal(const FunctionInfo &info) const;
  bool hasProtocolFormal(const FunctionInfo &info) const;
  bool hasClassFormal(const FunctionInfo &info) const;
  bool hasDefaultForFormal(const FunctionInfo &info,
                           std::size_t formalIndex) const;
  std::optional<FunctionInfo>
  specializeFunctionCall(const parser::Node &expr, const FunctionInfo &info,
                         const std::vector<parser::NodePtr> &args);
  std::optional<FunctionInfo>
  specializeProtocolFunctionCall(const parser::Node &expr,
                                 const FunctionInfo &info,
                                 const std::vector<parser::NodePtr> &args);
  std::optional<FunctionInfo>
  specializeClassFunctionCall(const parser::Node &expr,
                              const FunctionInfo &info,
                              const std::vector<parser::NodePtr> &args);
  std::optional<FunctionInfo>
  resolveCallableInfo(const parser::Node &expr) const;
  std::optional<FunctionInfo>
  resolveCallableArgInfo(const parser::Node &call, const FunctionInfo &info,
                         std::size_t formalIndex) const;
  std::optional<FunctionInfo> resolveCallableInfo(mlir::Value callable) const;
  std::optional<FunctionInfo>
  findCallableInfoBySymbol(llvm::StringRef symbolName) const;
  Value emitFunctionObject(const parser::Node &anchor,
                           const FunctionInfo &info);
  Value emitFunctionDefaults(const parser::Node &anchor,
                             const FunctionInfo &info);
  Value emitFunctionKwdefaults(const parser::Node &anchor,
                               const FunctionInfo &info);
  Value emitFunctionClosure(const parser::Node &anchor,
                            const FunctionInfo &info);
  void emitFunctionBinding(const parser::Node &function);
  std::optional<CallArgumentTuples> emitExplicitCallArgumentTuples(
      const parser::Node &expr, const FunctionInfo &info,
      const std::vector<parser::NodePtr> &args, std::size_t firstFormal = 0,
      llvm::ArrayRef<Value> leadingArgs = {});
  Value emitFromPrimCall(const parser::Node &expr,
                         const std::vector<parser::NodePtr> &args);
  Value emitToPrimCall(const parser::Node &expr,
                       const std::vector<parser::NodePtr> &args);
  Value emitClassConstructorCall(const parser::Node &expr, llvm::StringRef name,
                                 const std::vector<parser::NodePtr> &args);
  Value emitMethodCall(const parser::Node &expr, const parser::Node &func,
                       const std::vector<parser::NodePtr> &args);
  std::optional<FunctionInfo> resolveClassMethod(const parser::Node &anchor,
                                                 const Value &receiver,
                                                 llvm::StringRef methodName);
  std::optional<std::vector<Value>> prepareResolvedMethodCallArguments(
      const parser::Node &anchor, const Value &receiver,
      const FunctionInfo &method, llvm::ArrayRef<Value> userArgs);
  Value emitResolvedMethodCall(const parser::Node &anchor,
                               const Value &receiver,
                               const FunctionInfo &method,
                               llvm::ArrayRef<Value> userArgs);
  std::optional<SyncIteratorResolution>
  resolveSyncIterator(const parser::Node &anchor, Value iterable,
                      bool requireConcreteIterator);
  Value emitListMethodCall(const parser::Node &expr, const Value &receiver,
                           llvm::StringRef methodName,
                           const std::vector<parser::NodePtr> &args);
  Value emitTupleCountMethodCall(const parser::Node &expr,
                                 const Value &receiver,
                                 const std::vector<parser::NodePtr> &args);
  Value emitDictGetMethodCall(const parser::Node &expr, const Value &receiver,
                              const std::vector<parser::NodePtr> &args);
  Value emitExceptionCall(const parser::Node &expr, llvm::StringRef className,
                          const std::vector<parser::NodePtr> &args);
  std::optional<std::string> asyncioBuiltinName(const parser::Node &func);
  Value emitAsyncioCall(const parser::Node &expr, llvm::StringRef name,
                        const std::vector<parser::NodePtr> &args);
  Value emitAsyncioGather(const parser::Node &expr,
                          const std::vector<parser::NodePtr> &args);
  Value emitFunctionCall(const parser::Node &expr, const FunctionInfo &info,
                         const std::vector<parser::NodePtr> &args);
  CallArgumentTuples emitCallArgumentTuples(const FunctionInfo &info,
                                            const std::vector<Value> &actuals);
  Value emitMayThrowFunctionCall(const parser::Node &expr,
                                 const FunctionInfo &info, mlir::Value callee,
                                 Value posargs, Value kwnames, Value kwvalues);
  Value emitMayThrowCallableCall(const parser::Node &expr, mlir::Value callee,
                                 mlir::Type resultType, Value posargs,
                                 Value kwnames, Value kwvalues);
  Value emitListConstructorCall(const parser::Node &expr,
                                const std::vector<parser::NodePtr> &args);
  Value emitTupleConstructorCall(const parser::Node &expr,
                                 const std::vector<parser::NodePtr> &args);
  Value emitBoolConstructorCall(const parser::Node &expr,
                                const std::vector<parser::NodePtr> &args);
  Value emitIntConstructorCall(const parser::Node &expr,
                               const std::vector<parser::NodePtr> &args);
  Value emitFloatConstructorCall(const parser::Node &expr,
                                 const std::vector<parser::NodePtr> &args);
  std::optional<Value> emitProtocolUnaryConversionCall(
      const parser::Node &anchor, Value receiver, llvm::StringRef protocolName,
      llvm::StringRef methodName, mlir::Type resultType);
  Value emitStrConstructorCall(const parser::Node &expr,
                               const std::vector<parser::NodePtr> &args);
  Value emitReprCall(const parser::Node &expr,
                     const std::vector<parser::NodePtr> &args);
  Value emitIsinstanceCall(const parser::Node &expr);
  // Protocol oracle (rfc/iterator-protocol.md): the conformance closure of
  // the abstract container table over concrete types.
  std::optional<protocols::ProtocolMethod> resolveProtocolMethodContract(
      const parser::Node &anchor, mlir::Type receiverType,
      llvm::StringRef methodName, llvm::ArrayRef<mlir::Type> argumentTypes,
      llvm::StringRef contextLabel);
  std::optional<mlir::Type> resolveProtocolMethodResult(
      const parser::Node &anchor, mlir::Type receiverType,
      llvm::StringRef methodName, llvm::ArrayRef<mlir::Type> argumentTypes,
      llvm::StringRef contextLabel);
  py::CallableType unaryMethodContract(mlir::Type receiverType,
                                       mlir::Type argumentType,
                                       mlir::Type resultType);
  py::CallableType containsMethodContract(mlir::Type receiverType,
                                          mlir::Type itemType);
  std::optional<mlir::Type> protocolIterableElement(mlir::Type type);
  std::optional<mlir::Type> protocolAsyncIterableElement(mlir::Type type);
  Value emitIterCall(const parser::Node &expr,
                     const std::vector<parser::NodePtr> &args);
  Value emitNextValue(const parser::Node &anchor, Value iterator,
                      llvm::StringRef contextLabel);
  Value emitNextCall(const parser::Node &expr,
                     const std::vector<parser::NodePtr> &args);
  Value emitLenCall(const parser::Node &expr,
                    const std::vector<parser::NodePtr> &args);
  Value emitPrintCall(const parser::Node &expr,
                      const std::vector<parser::NodePtr> &args);
  Value emitRepr(const Value &input);
  Value emitTuple(const std::vector<Value> &elements);
  Value emitEmptyTuple();
};

} // namespace lython::emitter
