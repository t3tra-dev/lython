#pragma once

#include "Builder.h"
#include "TypeInference.h"

#include "PyDialectTypes.h"
#include "lython/parser/Ast.h"
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

struct Value {
  mlir::Value value;
  mlir::Type type;
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

struct FunctionInfo {
  std::string name;
  std::string symbolName;
  std::vector<std::string> argNames;
  std::vector<parser::NodePtr> defaultValues;
  std::size_t positionalOnlyCount = 0;
  std::size_t positionalCount = 0;
  std::vector<std::string> kwonlyNames;
  std::vector<parser::NodePtr> kwonlyDefaultValues;
  std::optional<std::string> varargName;
  mlir::Type varargType;
  std::optional<std::string> kwargName;
  mlir::Type kwargType;
  llvm::SmallVector<mlir::Type> argTypes;
  std::vector<ClosureCapture> closureCaptures;
  mlir::Type resultType;
  py::FuncSignatureType signatureType;
  py::FuncType functionType;
  mlir::FunctionType nativeFunctionType;
  mlir::FunctionType asyncFunctionType;
  bool isNative = false;
  bool isAsync = false;
  bool isInitMethod = false;
  bool mutatesSelf = false;
  bool mayThrow = false;
  bool isSpecialization = false;
  bool requiresCallableValue = false;
  std::optional<std::string> returnedCallableSymbolName;
  std::optional<std::size_t> returnedCallableArgIndex;
  std::shared_ptr<ReturnedCallableSummary> returnedCallable;
};

struct ReturnedCallableSummary {
  FunctionInfo info;
  std::vector<std::optional<std::size_t>> closureCaptureArgIndices;
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
};

struct ClassInfo {
  std::string name;
  std::vector<std::string> baseNames;
  std::vector<std::string> mro;
  std::map<std::string, mlir::Type> fields;
  std::map<std::string, std::string> methods;
  std::map<std::string, parser::NodePtr> methodNodes;
  std::map<std::string, parser::NodePtr> ownMethodNodes;
  bool inheritanceResolved = false;
};

struct PrimitiveConstant {
  mlir::Type type;
  std::int64_t integerValue = 0;
  double floatValue = 0.0;
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
  std::map<std::string, FunctionSpecialization> functionSpecializations;
  std::set<std::string> emittedFunctionSpecializations;
  std::map<std::string, ClassInfo> classes;
  std::map<std::string, mlir::Type> typeAliases;
  std::map<std::string, TypeAliasInfo> genericTypeAliases;
  std::map<std::string, PrimitiveConstant> primitiveConstants;
  std::set<std::string> activeGlobalNames;
  std::set<std::string> activeNonlocalNames;
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
  py::CoroutineType coroutineType(mlir::Type resultType);
  py::TaskType taskType(mlir::Type resultType);
  py::FutureType futureType(mlir::Type resultType);
  py::DictType dictType(mlir::Type keyType, mlir::Type valueType);
  py::ListType listType(mlir::Type elementType);
  mlir::Type i32Type();
  mlir::Type i1Type();

  EmitResult finish();
  void error(const parser::Node &node, std::string message);

  mlir::ArrayAttr stringArrayAttr(llvm::ArrayRef<std::string> values);
  mlir::ArrayAttr typeArrayAttr(llvm::ArrayRef<mlir::Type> values);
  std::optional<mlir::Type> typeFromAnnotation(const parser::NodePtr &node);
  std::optional<mlir::Type>
  typeFromAnnotation(const parser::NodePtr &node,
                     const std::map<std::string, mlir::Type> &typeVariables,
                     std::set<std::string> &activeAliases);
  bool isTypeAliasMarker(const parser::NodePtr &node) const;
  bool typeAssignable(mlir::Type expected, mlir::Type actual);
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
  py::FuncSignatureType
  functionSignatureType(llvm::ArrayRef<mlir::Type> argTypes,
                        mlir::Type resultType, mlir::Type varargType = {},
                        llvm::ArrayRef<mlir::Type> kwonlyTypes = {},
                        mlir::Type kwargType = {});
  mlir::FunctionType asyncFunctionType(llvm::ArrayRef<mlir::Type> argTypes,
                                       mlir::Type resultType);
  mlir::Type awaitablePayloadType(mlir::Type type);
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

  std::optional<FunctionInfo> parseFunctionInfo(const parser::Node &function);
  std::optional<FunctionInfo> parseMethodInfo(const parser::Node &function,
                                              llvm::StringRef className);
  std::optional<FunctionInfo> parseLambdaInfo(const parser::Node &lambda,
                                              py::FuncType expectedType);
  std::optional<mlir::Type> parameterType(const parser::Node &arg,
                                          mlir::Type fallbackType = {});
  bool appendAnnotatedParameter(const parser::Node &arg, FunctionInfo &info,
                                llvm::StringRef role,
                                mlir::Type fallbackType = {});
  bool collectCallableDefaults(const parser::Node &callable,
                               const parser::Node &arguments,
                               FunctionInfo &info);
  std::optional<std::vector<std::string>>
  computeClassMro(llvm::StringRef className, const parser::Node &anchor,
                  std::set<std::string> &visiting);
  bool resolveClassInheritance(llvm::StringRef className,
                               const parser::Node &anchor,
                               std::set<std::string> &resolving);
  void scanClasses(const parser::Node &moduleNode);
  void scanTypeAliases(const parser::Node &moduleNode);
  void scanFunctions(const parser::Node &moduleNode);
  std::optional<FunctionTypeComment>
  parseFunctionTypeComment(const parser::Node &function);
  bool expressionMayThrow(const parser::Node &expr) const;
  bool statementMayThrow(const parser::Node &stmt) const;
  bool
  statementListMayThrow(const std::vector<parser::NodePtr> &statements) const;
  void propagateMayThrow(const parser::Node &moduleNode);
  py::ClassOp createClass(llvm::StringRef name, mlir::ArrayAttr baseNames = {});
  py::FuncOp createFunc(llvm::StringRef name, py::FuncSignatureType signature,
                        mlir::ArrayAttr argNames = {}, bool hasVararg = false,
                        bool hasKwarg = false, bool mayThrow = false,
                        mlir::ArrayAttr kwonlyNames = {},
                        mlir::ArrayAttr closureTypes = {});
  void addEntryBlock(py::FuncOp func, llvm::ArrayRef<mlir::Type> argTypes);
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
  void emitClassDefs(const parser::Node &moduleNode);
  void emitClassDef(const parser::Node &classNode);
  void emitMethodDef(const parser::Node &function, const FunctionInfo &info);

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
  void emitWith(const parser::Node &stmt);
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
  std::optional<std::size_t> staticLength(const parser::Node &anchor,
                                          const Value &value,
                                          llvm::StringRef context);
  std::optional<std::vector<Value>>
  finiteSequenceElements(const parser::Node &anchor, const Value &sequence,
                         llvm::StringRef context);
  Value emitTupleLiteral(const parser::Node &expr,
                         mlir::Type expectedTupleType = {});
  Value emitSubscript(const parser::Node &expr);
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
  bool hasDefaultForFormal(const FunctionInfo &info,
                           std::size_t formalIndex) const;
  std::optional<FunctionInfo>
  specializeFunctionCall(const parser::Node &expr, const FunctionInfo &info,
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
  Value emitResolvedMethodCall(const parser::Node &anchor,
                               const Value &receiver,
                               const FunctionInfo &method,
                               llvm::ArrayRef<Value> userArgs);
  Value emitListMethodCall(const parser::Node &expr, const Value &receiver,
                           llvm::StringRef methodName,
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
  Value emitStrConstructorCall(const parser::Node &expr,
                               const std::vector<parser::NodePtr> &args);
  Value emitReprCall(const parser::Node &expr,
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
