#pragma once

// Internal implementation surface for the runtime bundle lowering pass.

#include "Runtime/Manifest/Index.h"
#include "Runtime/Model/Bundles.h"

#include "PyDialectTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::runtime_lowering {

class RuntimeBundleLowerer {
public:
  explicit RuntimeBundleLowerer(mlir::ModuleOp module);
  mlir::LogicalResult lowerModule();

private:
  struct EmittedRuntimeCall {
    RuntimeSymbol symbol;
    mlir::func::CallOp call;
  };

  const RuntimeValueShape *runtimeValueShapeFor(mlir::Operation *op,
                                                mlir::Type type,
                                                llvm::StringRef purpose) const;
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
  runtimeValueTypesFor(mlir::Operation *op, mlir::Type type,
                       llvm::StringRef purpose) const;
  py::ClassOp classForContract(mlir::Type type) const;
  bool classDefinesMethod(mlir::Type type, llvm::StringRef name) const;
  llvm::SmallVector<mlir::Type, 8>
  classFieldContractTypes(py::ClassOp classOp) const;
  std::optional<unsigned> classFieldIndex(py::ClassOp classOp,
                                          llvm::StringRef name) const;
  mlir::FailureOr<unsigned>
  classFieldValueOffset(mlir::Operation *op, py::ClassOp classOp,
                        unsigned fieldIndex, llvm::StringRef purpose) const;
  std::optional<unsigned> findUnionMemberIndex(py::UnionType unionType,
                                               mlir::Type member) const;
  mlir::FailureOr<unsigned>
  requireUnionMemberIndex(mlir::Operation *op, py::UnionType unionType,
                          mlir::Type member, llvm::StringRef purpose) const;
  mlir::FailureOr<unsigned>
  unionMemberValueOffset(mlir::Operation *op, py::UnionType unionType,
                         unsigned memberIndex, llvm::StringRef purpose) const;
  mlir::LogicalResult
  appendUnionRuntimeValues(mlir::Operation *op, py::UnionType resultUnion,
                           const RuntimeBundle &source, mlir::Type sourceType,
                           llvm::SmallVectorImpl<mlir::Value> &values);
  mlir::LogicalResult
  appendRuntimeValueTypes(mlir::Operation *op, mlir::Type type,
                          llvm::SmallVectorImpl<mlir::Type> &types) const;
  bool hasPrimitiveI64ABI(mlir::Type type) const;
  void appendPrimitiveI64EvidenceTypes(
      mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &types) const;
  mlir::LogicalResult appendPrimitiveI64EvidenceOperand(
      mlir::Operation *op, mlir::FunctionType functionType,
      unsigned &inputIndex, const RuntimeBundle &source,
      llvm::SmallVectorImpl<mlir::Value> &operands);
  llvm::SmallVector<mlir::Type, 4>
  callableClosureTypes(mlir::func::FuncOp function) const;
  mlir::Type callableVarargValueType(mlir::func::FuncOp function,
                                     py::CallableType callable) const;
  mlir::Type callableKwargValueType(mlir::func::FuncOp function,
                                    py::CallableType callable) const;
  llvm::SmallVector<mlir::Type, 8>
  callableLogicalInputTypes(mlir::func::FuncOp function,
                            py::CallableType callable) const;
  static mlir::Value stripReturnedObjectView(mlir::Value value);
  mlir::LogicalResult buildReturnedValueSummaries();
  mlir::LogicalResult buildReturnedCallableSummaries();
  mlir::LogicalResult buildReturnedCoroutineSummaries();
  mlir::LogicalResult buildReturnedObjectEvidenceSummaries();
  mlir::LogicalResult buildCallableProtocolArgumentABIs();
  mlir::LogicalResult buildCallableArgumentEvidenceABIs();
  mlir::LogicalResult buildCallableAggregateEvidenceABIs();
  mlir::LogicalResult buildPrimitiveI64CallableClones();
  mlir::LogicalResult prepareCallableFunctionABIs();
  bool isPrimitiveI64CallableClone(mlir::func::FuncOp function) const;
  bool isPrimitiveI64CallableEligible(mlir::func::FuncOp function) const;
  std::optional<std::string> primitiveI64CloneFor(llvm::StringRef target) const;
  mlir::LogicalResult seedPrimitiveI64CallableEntryArgumentBundles(
      mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes);
  mlir::LogicalResult seedCallableEntryArgumentBundles(
      mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes,
      mlir::ArrayRef<mlir::Type> abiTypes,
      const CallableAggregateEvidenceABI *aggregateEvidence);
  mlir::LogicalResult validateObjectShape(mlir::Operation *op,
                                          mlir::Type contract,
                                          mlir::ValueRange values) const;
  mlir::LogicalResult makeObjectBundle(mlir::Operation *op, mlir::Type contract,
                                       mlir::ValueRange values,
                                       RuntimeBundle &bundle) const;
  mlir::LogicalResult makePrimitiveI64Bundle(mlir::Operation *op,
                                             mlir::Type contract,
                                             mlir::Value value,
                                             mlir::Value valid,
                                             RuntimeBundle &bundle) const;
  void seedPrimitiveI64Evidence(mlir::Operation *op, mlir::Type contract,
                                mlir::ValueRange rawValues,
                                RuntimeBundle &bundle);
  bool hasLazyPrimitiveI64Object(const RuntimeBundle &bundle) const;
  bool canMaterializePrimitiveI64Object(const RuntimeBundle &bundle) const;
  bool hasPrimitiveI64Evidence(const RuntimeBundle *bundle) const;
  bool allSourcesHavePrimitiveI64Evidence(
      llvm::ArrayRef<const RuntimeBundle *> sources) const;
  bool isStaticCtypesBinding(llvm::StringRef binding) const;
  bool isStaticCtypesModuleBinding(llvm::StringRef binding) const;
  bool isStaticCtypesCallable(llvm::StringRef binding) const;
  bool isErasedCtypesContract(llvm::StringRef contract) const;
  bool isStaticCtypesLibraryContract(llvm::StringRef contract) const;
  mlir::LogicalResult lowerStaticCtypesBindingRef(py::BindingRefOp op);
  mlir::LogicalResult lowerStaticCtypesModuleBindingRef(py::BindingRefOp op);
  mlir::LogicalResult
  lowerStaticCtypesModuleAttrGet(py::AttrGetOp op, const RuntimeBundle &object);
  mlir::LogicalResult
  lowerStaticCtypesValueAttrGet(py::AttrGetOp op, const RuntimeBundle &object);
  mlir::LogicalResult
  lowerStaticCtypesFieldDescriptorAttrGet(py::AttrGetOp op,
                                          const RuntimeBundle &object);
  mlir::LogicalResult
  lowerStaticCtypesTypeFieldDescriptorGet(py::AttrGetOp op,
                                          const RuntimeBundle &object);
  mlir::LogicalResult
  lowerStaticCtypesFieldAttrGet(py::AttrGetOp op, const RuntimeBundle &object);
  mlir::LogicalResult lowerStaticCtypesFieldAttrSet(py::AttrSetOp op,
                                                    const RuntimeBundle &object,
                                                    const RuntimeBundle *value);
  mlir::LogicalResult lowerStaticCtypesGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index);
  mlir::LogicalResult lowerStaticCtypesModuleCall(py::CallOp op,
                                                  const RuntimeBundle &receiver,
                                                  llvm::StringRef methodName);
  mlir::LogicalResult
  lowerStaticCtypesTypeObjectCall(py::CallOp op, const RuntimeBundle &callable);
  mlir::LogicalResult lowerStaticCtypesTypeObjectMethodCall(
      py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName);
  mlir::LogicalResult lowerStaticCtypesArrayTypeMul(mlir::Operation *op,
                                                    const RuntimeBundle &lhs,
                                                    const RuntimeBundle &rhs,
                                                    mlir::Value resultValue);
  mlir::LogicalResult bindErasedCtypesNew(py::NewOp op,
                                          llvm::StringRef contract);
  mlir::LogicalResult bindStaticCtypesLibraryNew(py::NewOp op,
                                                 llvm::StringRef contract);
  mlir::LogicalResult
  lowerErasedCtypesInit(py::InitOp op, const RuntimeBundle &instance,
                        llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult
  lowerStaticCtypesLibraryInit(py::InitOp op, const RuntimeBundle &instance,
                               llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult lowerStaticCtypesAttrGet(py::AttrGetOp op,
                                               const RuntimeBundle &object);
  mlir::LogicalResult lowerStaticCtypesAttrSet(py::AttrSetOp op,
                                               const RuntimeBundle &object,
                                               const RuntimeBundle *value);
  mlir::LogicalResult lowerStaticCtypesCall(py::CallOp op,
                                            const RuntimeBundle &callable);
  mlir::LogicalResult
  lowerStaticCtypesNativeCall(py::CallOp op, const RuntimeBundle &callable);
  mlir::FailureOr<RuntimeValue>
  materializePrimitiveI64Object(mlir::Operation *op,
                                const RuntimeBundle &bundle);
  bool objectShapeMatches(llvm::StringRef contract,
                          mlir::ValueRange values) const;
  bool isBuiltinsObjectHeaderType(mlir::Type type) const;
  bool isErasedObjectStorageType(mlir::Type type) const;
  mlir::FailureOr<mlir::Value> objectHeaderView(mlir::Operation *op,
                                                const RuntimeValue &value);
  mlir::FailureOr<mlir::Value>
  erasedObjectStorageView(mlir::Operation *op, const RuntimeValue &value,
                          mlir::Type targetType);
  bool rawValuesMatchRuntimeInputs(const RuntimeSymbol &symbol,
                                   mlir::ValueRange values) const;
  mlir::LogicalResult initializeObjectFromRawValues(mlir::Operation *op,
                                                    mlir::Type contract,
                                                    mlir::ValueRange values,
                                                    RuntimeBundle &bundle,
                                                    bool emitErrors = true);
  mlir::LogicalResult bundleRawObjectValues(mlir::Operation *op,
                                            mlir::Type contract,
                                            mlir::ValueRange values,
                                            RuntimeBundle &bundle);
  mlir::LogicalResult materializeDefaultValue(mlir::Operation *op,
                                              mlir::Type parameterType,
                                              mlir::Attribute attr,
                                              RuntimeBundle &bundle);
  mlir::FailureOr<mlir::Value> materializeDeadPhysicalValue(mlir::Operation *op,
                                                            mlir::Type type);
  mlir::FailureOr<RuntimeValue>
  materializeDeadObjectValue(mlir::Operation *op, mlir::Type contract,
                             llvm::StringRef purpose);
  mlir::LogicalResult materializeStringObject(mlir::Operation *op,
                                              llvm::StringRef text,
                                              RuntimeBundle &bundle);
  bool needsDefaultObjectRepr(const RuntimeBundle &object) const;
  mlir::LogicalResult materializeDefaultObjectRepr(mlir::Operation *op,
                                                   const RuntimeBundle &object,
                                                   RuntimeBundle &bundle);
  mlir::LogicalResult assignObjectBundle(mlir::Operation *op, mlir::Value value,
                                         mlir::Type contract,
                                         mlir::ValueRange values);
  mlir::LogicalResult bindEvidenceObjectResult(mlir::Operation *op,
                                               mlir::Value resultValue,
                                               llvm::StringRef label,
                                               const RuntimeValue &value);
  mlir::LogicalResult bindSelectedEvidenceObjectResult(mlir::Operation *op,
                                                       mlir::Value resultValue,
                                                       RuntimeBundle bundle);
  mlir::FailureOr<RuntimeBundle>
  selectEvidenceObjectByMatch(mlir::Operation *op, mlir::Value resultValue,
                              llvm::ArrayRef<RuntimeValue> candidates,
                              mlir::ValueRange matches, llvm::StringRef label,
                              llvm::StringRef missingContract,
                              llvm::StringRef missingMessage);
  mlir::FailureOr<RuntimeBundle> selectEvidenceObjectMiss(
      mlir::Operation *op, mlir::Value resultValue,
      llvm::ArrayRef<RuntimeValue> candidates, llvm::StringRef label,
      llvm::StringRef missingContract, llvm::StringRef missingMessage);
  mlir::FailureOr<llvm::StringRef>
  requireMethodTarget(mlir::Operation *op, mlir::FlatSymbolRefAttr target,
                      llvm::StringRef expectedName) const;

  template <typename Op> mlir::LogicalResult lowerAliasViewOp(Op op) {
    return lowerAliasView(op.getOperation(), op.getInput(), op.getResult());
  }

  template <typename Op> mlir::LogicalResult lowerUnaryMethodOp(Op op) {
    mlir::FailureOr<llvm::StringRef> methodName = requireMethodTarget(
        op.getOperation(), op.getTargetAttr(), op.getMethodName());
    if (mlir::failed(methodName))
      return mlir::failure();
    return lowerUnarySpecial(op.getOperation(), op.getInput(), *methodName,
                             op.getResult());
  }

  template <typename Op> mlir::LogicalResult lowerBinaryMethodOp(Op op) {
    mlir::FailureOr<llvm::StringRef> methodName = requireMethodTarget(
        op.getOperation(), op.getTargetAttr(), op.getMethodName());
    if (mlir::failed(methodName))
      return mlir::failure();
    return lowerBinarySpecial(op.getOperation(), op.getLhs(), op.getRhs(),
                              *methodName, op.getResult());
  }

  template <typename Op>
  mlir::LogicalResult lowerNamedUnaryMethodOp(Op op,
                                              llvm::StringRef methodName) {
    mlir::FailureOr<llvm::StringRef> target =
        requireMethodTarget(op.getOperation(), op.getTargetAttr(), methodName);
    if (mlir::failed(target))
      return mlir::failure();
    return lowerUnarySpecial(op.getOperation(), op.getInput(), *target,
                             op.getResult());
  }

  mlir::LogicalResult lowerPyOp(mlir::Operation *op);
  mlir::LogicalResult lowerStructuredTryOps();
  mlir::LogicalResult lowerTry(py::TryOp op);
  mlir::LogicalResult lowerStrConstant(py::StrConstantOp op);
  bool isStaticKeywordName(py::StrConstantOp op) const;
  mlir::LogicalResult lowerIntConstant(py::IntConstantOp op);
  mlir::LogicalResult lowerFloatConstant(py::FloatConstantOp op);
  mlir::LogicalResult lowerBoolConstant(py::BoolConstantOp op);
  mlir::LogicalResult lowerNone(py::NoneOp op);
  mlir::LogicalResult lowerCastFromPrim(py::CastFromPrimOp op);
  mlir::LogicalResult lowerUnionWrap(py::UnionWrapOp op);
  mlir::LogicalResult lowerUnionTest(py::UnionTestOp op);
  mlir::LogicalResult lowerUnionUnwrap(py::UnionUnwrapOp op);
  mlir::LogicalResult lowerTypeObject(py::TypeObjectOp op);
  mlir::LogicalResult lowerAttrGet(py::AttrGetOp op);
  mlir::LogicalResult lowerAttrSet(py::AttrSetOp op);
  mlir::LogicalResult lowerPack(py::PackOp op);
  mlir::LogicalResult lowerBindingRef(py::BindingRefOp op);
  mlir::LogicalResult lowerFunctionBindingRef(py::BindingRefOp op,
                                              mlir::func::FuncOp function);
  mlir::LogicalResult appendClosureValues(py::BindingRefOp op,
                                          mlir::func::FuncOp function,
                                          RuntimeBundle &bundle);
  mlir::LogicalResult lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue);
  mlir::LogicalResult collectObjectSources(
      mlir::Operation *op, mlir::ValueRange values, llvm::StringRef message,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources) const;
  mlir::LogicalResult collectPackedObjectSources(
      mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> *unpackedSources = nullptr) const;
  mlir::LogicalResult requireEmptyAggregate(mlir::Operation *op,
                                            mlir::Value packValue,
                                            llvm::StringRef label) const;
  mlir::LogicalResult verifySelectedRuntimeTarget(mlir::Operation *op,
                                                  RuntimeSymbol &symbol);
  mlir::FailureOr<RuntimeSymbol>
  selectManifestMethod(mlir::Operation *op, const RuntimeBundle &receiver,
                       llvm::StringRef methodName,
                       llvm::ArrayRef<const RuntimeBundle *> sources,
                       bool allowUnusedSources);
  mlir::LogicalResult emitManifestMethodCall(
      mlir::Operation *op, const RuntimeBundle &receiver,
      llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
      bool allowUnusedSources, std::optional<EmittedRuntimeCall> &emitted);
  std::string resultContractFor(mlir::Value resultValue,
                                const RuntimeSymbol &symbol,
                                bool preferManifestObjectResult) const;
  mlir::LogicalResult
  bindRuntimeCallResult(mlir::Operation *op, mlir::Value resultValue,
                        const EmittedRuntimeCall &emitted,
                        bool preferManifestObjectResult = false,
                        const RuntimeBundle *receiverEvidence = nullptr);
  mlir::LogicalResult lowerManifestMethodResult(
      mlir::Operation *op, mlir::Value resultValue,
      const RuntimeBundle &receiver, llvm::StringRef methodName,
      llvm::ArrayRef<const RuntimeBundle *> sources, bool allowUnusedSources,
      bool preferManifestObjectResult = false);
  mlir::LogicalResult lowerManifestI1MethodResult(
      mlir::Operation *op, mlir::Value resultValue,
      const RuntimeBundle &receiver, llvm::StringRef methodName,
      llvm::ArrayRef<const RuntimeBundle *> sources, bool allowUnusedSources);
  mlir::LogicalResult
  lowerManifestVoidMethod(mlir::Operation *op, const RuntimeBundle &receiver,
                          llvm::StringRef methodName,
                          llvm::ArrayRef<const RuntimeBundle *> sources,
                          bool allowUnusedSources);
  mlir::LogicalResult lowerNew(py::NewOp op);
  mlir::LogicalResult lowerInit(py::InitOp op);
  mlir::LogicalResult emitRuntimeException(mlir::Operation *op,
                                           llvm::StringRef contract,
                                           llvm::StringRef message);
  mlir::LogicalResult
  emitRuntimeExceptionFromMessageObject(mlir::Operation *op,
                                        llvm::StringRef contract,
                                        const RuntimeBundle &messageObject);
  mlir::LogicalResult lowerRaise(py::RaiseOp op);
  mlir::LogicalResult lowerRaiseCurrent(py::RaiseCurrentOp op);
  mlir::LogicalResult lowerExceptMatch(py::ExceptMatchOp op);
  mlir::LogicalResult lowerExceptCurrentMatch(py::ExceptCurrentMatchOp op);
  mlir::LogicalResult emitTracebackFrame(mlir::Operation *op);
  mlir::LogicalResult lowerCall(py::CallOp op);
  mlir::LogicalResult lowerBoundMethodCall(py::CallOp op,
                                           const RuntimeBundle &receiver,
                                           llvm::StringRef methodName);
  mlir::LogicalResult lowerFutureResultEvidence(mlir::Operation *op,
                                                mlir::Value resultValue,
                                                const RuntimeBundle &receiver,
                                                llvm::StringRef label);
  mlir::LogicalResult lowerAsyncioSleepEvidenceAwait(mlir::Operation *op,
                                                     mlir::Value resultValue,
                                                     RuntimeBundle &awaitable,
                                                     llvm::StringRef label);
  mlir::LogicalResult lowerFutureBoundMethod(py::CallOp op,
                                             RuntimeBundle &receiver,
                                             llvm::StringRef methodName);
  mlir::LogicalResult lowerAsyncioSleepCall(py::CallOp op,
                                            const RuntimeSymbol &symbol);
  mlir::LogicalResult lowerObjectCallableCall(py::CallOp op,
                                              const RuntimeBundle &callable);
  mlir::LogicalResult lowerFunctionTargetCall(py::CallOp op,
                                              const RuntimeBundle &callable);
  mlir::LogicalResult
  lowerPrimitiveI64CloneCall(py::CallOp op, mlir::func::FuncOp target,
                             llvm::StringRef targetName,
                             llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult lowerPrimitiveI64CloneFallbackCall(
      py::CallOp op, mlir::func::FuncOp original, llvm::StringRef originalName,
      mlir::func::FuncOp clone, llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult emitPrimitiveI64CloneFallbackResult(
      py::CallOp op, mlir::func::FuncOp original, llvm::StringRef originalName,
      mlir::func::FuncOp clone, llvm::ArrayRef<const RuntimeBundle *> sources,
      RuntimeBundle &result);
  mlir::LogicalResult
  lowerIndirectFunctionObjectCall(py::CallOp op, const RuntimeBundle &callable);
  llvm::SmallVector<mlir::func::FuncOp, 8>
  collectIndirectCallableTargets(py::CallOp op, const RuntimeBundle &callable);
  mlir::LogicalResult collectFunctionTargetRuntimeSources(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      const RuntimeBundle &callable,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
      llvm::SmallVectorImpl<RuntimeBundle> &closureSources,
      llvm::SmallVectorImpl<RuntimeBundle> &argumentEvidenceSources,
      llvm::SmallVectorImpl<RuntimeBundle> &aggregateEvidenceSources);
  mlir::LogicalResult appendIndirectCallableResultOperands(
      mlir::Operation *op, const RuntimeBundle &result,
      llvm::ArrayRef<mlir::Type> expectedTypes,
      llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::LogicalResult appendCallableAggregateEvidenceSources(
      py::CallOp op, llvm::StringRef targetName,
      const CallableAggregateEvidenceABI &evidence,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> &evidenceSources);
  mlir::LogicalResult appendCallableArgumentEvidenceSources(
      py::CallOp op, llvm::StringRef targetName,
      const CallableArgumentEvidenceABI &evidence,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> &evidenceSources);
  mlir::FailureOr<mlir::func::CallOp>
  emitFunctionTargetRuntimeCall(py::CallOp op, mlir::func::FuncOp target,
                                llvm::StringRef targetName,
                                llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult bundleFunctionTargetCallResult(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      mlir::func::CallOp call, llvm::ArrayRef<const RuntimeBundle *> sources,
      RuntimeBundle &result);
  mlir::LogicalResult
  bundlePrimitiveI64CloneCallResult(py::CallOp op, mlir::func::FuncOp target,
                                    mlir::func::CallOp call,
                                    RuntimeBundle &result);
  mlir::LogicalResult
  lowerAsyncFunctionTargetCall(py::CallOp op, mlir::func::FuncOp target,
                               llvm::StringRef targetName,
                               llvm::ArrayRef<const RuntimeBundle *> sources);
  std::optional<StaticCallableInvocation>
  collectStaticCallableInvocation(py::CallOp op) const;
  std::optional<CallableArgumentPlan>
  collectCallableArgumentPlan(py::CallOp op, py::CallableType callable,
                              bool emitErrors = false) const;
  std::optional<CallableAggregateEvidenceCall>
  collectCallableAggregateEvidence(py::CallOp op,
                                   py::CallableType callable) const;
  std::optional<llvm::SmallVector<mlir::Type, 4>>
  collectCallableArgumentSourceTypes(py::CallOp op,
                                     py::CallableType callable) const;
  mlir::LogicalResult collectFunctionCallSources(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults);
  mlir::LogicalResult materializeDefaultArgument(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      unsigned index, mlir::Type parameterType,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
      const RuntimeBundle *&source);
  mlir::LogicalResult
  materializeArityObject(mlir::Operation *op, mlir::Type contract,
                         std::uint64_t arity, RuntimeBundle &bundle,
                         mlir::ArrayRef<RuntimeValue> elements = {},
                         llvm::ArrayRef<std::string> keys = {});
  std::optional<std::string> keywordNameFromValue(mlir::Value value) const;
  mlir::LogicalResult lowerReceiverMethodResult(
      mlir::Operation *op, mlir::Value receiverValue, mlir::Value resultValue,
      llvm::StringRef missingSubject, llvm::StringRef methodName,
      bool preferManifestObjectResult = false);
  mlir::LogicalResult lowerBool(py::BoolOp op);
  mlir::LogicalResult lowerLen(py::LenOp op);
  mlir::FailureOr<bool>
  lowerSequenceEvidenceGetItem(py::GetItemOp op, const RuntimeBundle &container,
                               const RuntimeBundle &index);
  mlir::FailureOr<bool> lowerDictEvidenceGetItem(py::GetItemOp op,
                                                 const RuntimeBundle &container,
                                                 const RuntimeBundle &index);
  mlir::LogicalResult lowerGetItem(py::GetItemOp op);
  mlir::LogicalResult lowerSetItem(py::SetItemOp op);
  mlir::LogicalResult lowerDelItem(py::DelItemOp op);
  mlir::LogicalResult lowerContains(py::ContainsOp op);
  mlir::LogicalResult lowerIter(py::IterOp op);
  mlir::LogicalResult lowerNext(py::NextOp op);
  mlir::LogicalResult lowerEnter(py::EnterOp op);
  mlir::LogicalResult lowerExit(py::ExitOp op);
  mlir::LogicalResult lowerAEnter(py::AEnterOp op);
  mlir::LogicalResult lowerAExit(py::AExitOp op);
  mlir::LogicalResult lowerAIter(py::AIterOp op);
  mlir::LogicalResult lowerANext(py::ANextOp op);
  mlir::LogicalResult lowerAwait(py::AwaitOp op);
  mlir::LogicalResult lowerRound(py::RoundOp op);
  mlir::LogicalResult lowerUnarySpecial(mlir::Operation *op, mlir::Value input,
                                        llvm::StringRef methodName,
                                        mlir::Value resultValue);
  bool canAppendExactValueSequence(mlir::FunctionType functionType,
                                   unsigned inputIndex,
                                   const RuntimeBundle &source) const;
  mlir::LogicalResult
  appendRuntimeSource(mlir::Operation *op, const RuntimeSymbol &symbol,
                      mlir::FunctionType functionType, unsigned &inputIndex,
                      const RuntimeBundle &source,
                      llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::LogicalResult
  appendRuntimeSourceAs(mlir::Operation *op, const RuntimeSymbol &symbol,
                        mlir::FunctionType functionType, unsigned &inputIndex,
                        const RuntimeBundle &source, mlir::Type expected,
                        llvm::SmallVectorImpl<mlir::Value> &operands);
  bool canAppendRuntimeSource(const RuntimeSymbol &symbol,
                              mlir::FunctionType functionType,
                              unsigned &inputIndex,
                              const RuntimeBundle &source) const;
  mlir::LogicalResult appendImplicitRuntimeArgument(
      mlir::Operation *op, const RuntimeSymbol &symbol, unsigned &inputIndex,
      llvm::SmallVectorImpl<mlir::Value> &operands);
  bool canAppendImplicitRuntimeArgument(const RuntimeSymbol &symbol,
                                        unsigned &inputIndex) const;
  bool
  canBuildRuntimeCallOperands(const RuntimeSymbol &symbol,
                              llvm::ArrayRef<const RuntimeBundle *> sources,
                              bool allowUnusedSources,
                              const RuntimeBundle *classObject = nullptr) const;
  mlir::LogicalResult
  buildRuntimeCallOperands(mlir::Operation *op, const RuntimeSymbol &symbol,
                           llvm::ArrayRef<const RuntimeBundle *> sources,
                           llvm::SmallVectorImpl<mlir::Value> &operands,
                           bool allowUnusedSources,
                           const RuntimeBundle *classObject = nullptr);
  mlir::LogicalResult lowerBinarySpecial(mlir::Operation *op, mlir::Value lhs,
                                         mlir::Value rhs,
                                         llvm::StringRef methodName,
                                         mlir::Value resultValue);
  mlir::LogicalResult lowerPrimitiveI64BinarySpecial(
      mlir::Operation *op, llvm::StringRef methodName,
      llvm::ArrayRef<const RuntimeBundle *> sources, mlir::Value resultValue);
  mlir::LogicalResult
  collectSingleBuiltinArgument(py::CallOp op, const RuntimeSymbol &symbol,
                               const RuntimeBundle *&argument) const;
  mlir::LogicalResult lowerBuiltinMethodCall(py::CallOp op,
                                             const RuntimeSymbol &symbol);
  mlir::LogicalResult lowerBuiltinMethodSinkCall(py::CallOp op,
                                                 const RuntimeSymbol &symbol);
  mlir::LogicalResult lowerDirectBuiltinCall(py::CallOp op,
                                             const RuntimeSymbol &symbol);
  mlir::LogicalResult bundleRuntimeResults(mlir::Operation *op,
                                           mlir::Type expectedContract,
                                           mlir::func::CallOp call,
                                           RuntimeBundle &result);
  mlir::LogicalResult bundleRuntimeResults(mlir::Operation *op,
                                           mlir::Type expectedContract,
                                           mlir::ValueRange values,
                                           RuntimeBundle &result);
  mlir::LogicalResult
  appendBundlePhysicalOperands(mlir::Operation *op, const RuntimeBundle &bundle,
                               mlir::ArrayRef<mlir::Type> expectedTypes,
                               llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::LogicalResult ensureValueBundle(mlir::Operation *op, mlir::Value value);
  mlir::LogicalResult ensureOperationOperandBundles(mlir::Operation *op);
  mlir::LogicalResult
  lowerControlFlowBlockArgument(mlir::Operation *op,
                                mlir::BlockArgument argument);
  mlir::LogicalResult dropControlFlowLogicalBranchOperands();
  mlir::LogicalResult eraseControlFlowLogicalBlockArguments();
  const RuntimeBundle *bundleFor(mlir::Value value) const;
  mlir::Value materializeByteBuffer(mlir::Location loc, llvm::StringRef text);
  std::optional<std::int64_t> currentTryHandlerId() const;
  void emitTryCallSiteMarker(mlir::Location loc, std::int64_t id);
  void emitTryCallSiteMarkerIfNeeded(mlir::Location loc);
  mlir::func::FuncOp getOrCreateTryCallSiteMarker();
  mlir::func::FuncOp getOrCreateTryCatchMarker();
  mlir::func::FuncOp getOrCreateTryCatchAnchor();
  mlir::func::CallOp createRuntimeCall(mlir::Location loc,
                                       const RuntimeSymbol &symbol,
                                       mlir::ValueRange operands);
  std::int64_t functionTargetId(llvm::StringRef target);
  mlir::LogicalResult lowerFunctionReturns();
  mlir::LogicalResult eraseCallableLogicalEntryArgs();
  mlir::LogicalResult eraseLoweredPyOps();

  mlir::ModuleOp module;
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  RuntimeManifestIndex manifest;
  llvm::DenseMap<mlir::Value, RuntimeBundle> valueBundles;
  llvm::StringMap<ReturnedValueSummary> returnedValueSummaries;
  llvm::StringMap<ReturnedCallableSummary> returnedCallableSummaries;
  llvm::StringMap<ReturnedCoroutineSummary> returnedCoroutineSummaries;
  llvm::StringMap<ReturnedObjectEvidenceSummary>
      returnedObjectEvidenceSummaries;
  llvm::StringMap<llvm::SmallVector<mlir::Type, 8>>
      callableProtocolArgumentABIs;
  llvm::StringMap<CallableArgumentEvidenceABI> callableArgumentEvidenceABIs;
  llvm::StringMap<CallableAggregateEvidenceABI> callableAggregateEvidenceABIs;
  llvm::StringMap<std::string> primitiveI64CallableClones;
  llvm::StringMap<std::int64_t> functionTargetIds;
  llvm::DenseMap<mlir::Block *, std::int64_t> tryHandlerIds;
  llvm::SmallVector<CallableLogicalEntryArgs, 8> callableLogicalEntryArgCounts;
  llvm::SmallVector<ControlFlowLogicalBlockArgumentABI, 16>
      controlFlowLogicalBlockArguments;
  llvm::DenseSet<mlir::Value> controlFlowLogicalBlockArgumentSet;
  llvm::DenseSet<mlir::Value> controlFlowBlockArgumentsInProgress;
  std::int64_t nextFunctionTargetId = 1;
  std::int64_t nextTryHandlerId = 1;
  llvm::SmallVector<mlir::Operation *, 32> erase;
};

} // namespace py::runtime_lowering
