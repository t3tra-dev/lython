#pragma once

// Internal implementation surface for the runtime bundle lowering pass.
//
// Layers 1-2 of the transformation stack,
// as a DELIBERATE deviation from DialectConversion: per-op `lower*` methods
// (dispatched in Core/Dispatch.cpp) play the conversion-pattern role, and the
// physical ABI mapping (runtimeValueTypesFor + the RuntimeBundle expansion +
// the ABI rewrites under Runtime/ABI/) plays the TypeConverter role. Patterns
// do not fit here because a py op's lowering depends on evidence accumulated
// across ops in the bundle map, not on the op in isolation; legality is
// enforced by explicit earliest-boundary rejections and the inter-phase
// verifiers instead of a ConversionTarget.

#include "Ownership.h"
#include "Runtime/Manifest/Index.h"
#include "Runtime/Model/Bundles.h"

#include "PyDialectTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

namespace py::lowering {

class RuntimeBundleLowerer {
public:
  explicit RuntimeBundleLowerer(mlir::ModuleOp module);
  mlir::LogicalResult lowerModule();

private:
  enum class DeadObjectStorage { OwningHeap, StaticNonOwning };

  struct EmittedRuntimeCall {
    RuntimeSymbol symbol;
    mlir::func::CallOp call;
  };

  struct SourceGeneratorResumeResult {
    mlir::Value value;
    mlir::Value valid;
    mlir::Value hasValue;
  };

  struct CallableProtocolSpecialization {
    std::string cloneName;
    llvm::SmallVector<mlir::Type, 8> argumentTypes;
  };

  const RuntimeValueShape *runtimeValueShapeFor(mlir::Operation *op,
                                                mlir::Type type,
                                                llvm::StringRef purpose) const;
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
  runtimeValueTypesFor(mlir::Operation *op, mlir::Type type,
                       llvm::StringRef purpose) const;
  py::ClassOp classForContract(mlir::Type type) const;
  std::optional<std::int64_t> runtimeClassIdForClass(py::ClassOp classOp) const;
  std::optional<std::int64_t> runtimeClassIdForContract(mlir::Type type) const;
  mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>>
  runtimeClassIdsForNominalTarget(mlir::Operation *op,
                                  mlir::Type targetType) const;
  bool classDefinesMethod(mlir::Type type, llvm::StringRef name) const;
  std::optional<std::string> classMethodSymbol(py::ClassOp classOp,
                                               llvm::StringRef name) const;
  llvm::SmallVector<mlir::Type, 8>
  classFieldContractTypes(py::ClassOp classOp) const;
  std::optional<unsigned> classFieldIndex(py::ClassOp classOp,
                                          llvm::StringRef name) const;
  mlir::FailureOr<unsigned>
  classFieldValueOffset(mlir::Operation *op, py::ClassOp classOp,
                        unsigned fieldIndex, llvm::StringRef purpose) const;
  // Container-contract fields are stored BOX-FRONTED (one box16 slot; the
  // container's arrays live in the container's own box words) so in-place
  // mutation is an indirection through a stable box pointer — branch-, loop-
  // and realloc-safe. Erased-object fields share the same storage shape.
  bool classFieldStoredBoxed(mlir::Type fieldContract) const;
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
  classFieldStorageValueTypes(mlir::Operation *op, mlir::Type fieldContract,
                              llvm::StringRef purpose) const;
  mlir::LogicalResult writeBackFieldAlias(mlir::Operation *op,
                                          const RuntimeBundle &updatedField);
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
  mlir::LogicalResult buildReturnedStaticObjectSummaries();
  mlir::LogicalResult buildCallableProtocolArgumentABIs();
  mlir::LogicalResult buildCallableArgumentEvidenceABIs();
  mlir::LogicalResult buildCallableAggregateEvidenceABIs();
  mlir::LogicalResult buildPrimitiveI64CallableClones();
  mlir::LogicalResult prepareCallableFunctionABIs();
  bool isCallableProtocolTemplate(mlir::func::FuncOp function) const;
  std::optional<std::string> callableProtocolSpecializationFor(
      llvm::StringRef target,
      llvm::ArrayRef<const RuntimeBundle *> sources) const;
  mlir::FailureOr<mlir::func::FuncOp> selectCallableProtocolSpecialization(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult eraseCallableProtocolTemplateFunctions();
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
                                       RuntimeBundle &bundle,
                                       bool ownsObject = true) const;
  mlir::LogicalResult
  makeObjectBundleWithOwnership(mlir::Operation *op, mlir::Type contract,
                                mlir::ValueRange values, RuntimeBundle &bundle,
                                ownership::OwnershipKind ownership) const;
  mlir::LogicalResult markOwnedLocalObjectBundle(mlir::Operation *op,
                                                 mlir::Value logicalValue,
                                                 const RuntimeBundle &bundle);
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
  mlir::FailureOr<RuntimePrimitiveI64Evidence>
  emitPrimitiveI64ArithmeticEvidence(mlir::Operation *op,
                                     llvm::StringRef methodName,
                                     const RuntimePrimitiveI64Evidence &lhs,
                                     const RuntimePrimitiveI64Evidence &rhs);
  mlir::FailureOr<RuntimePrimitiveI64Evidence>
  materializeSourceGeneratorI64Value(
      mlir::Operation *op, mlir::Value value,
      llvm::ArrayRef<const RuntimeBundle *> frameSources,
      llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> &memo,
      std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence =
          std::nullopt);
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
  mlir::LogicalResult lowerGlobalGet(py::GlobalGetOp op);
  mlir::LogicalResult lowerObjectGlobalGet(py::GlobalGetOp op);
  mlir::LogicalResult lowerObjectGlobalSet(py::GlobalSetOp op);
  mlir::LLVM::GlobalOp moduleObjectGlobalCell(mlir::Operation *op,
                                              llvm::StringRef name,
                                              llvm::StringRef suffix);
  mlir::func::FuncOp globalViewFunction(mlir::Operation *op,
                                        mlir::Type element);
  mlir::Value loadObjectGlobalWord(mlir::Operation *op, llvm::StringRef name,
                                   llvm::StringRef suffix);
  void storeObjectGlobalWord(mlir::Operation *op, llvm::StringRef name,
                             llvm::StringRef suffix, mlir::Value word);
  mlir::LogicalResult
  loadObjectGlobalValues(mlir::Operation *op, llvm::StringRef name,
                         llvm::ArrayRef<mlir::Type> valueTypes,
                         llvm::SmallVectorImpl<mlir::Value> &values);
  mlir::LogicalResult lowerGlobalSet(py::GlobalSetOp op);
  // Process-lifetime i64 storage for a module-level int global, created on
  // first use. Reads/writes are plain load/store (async-signal-safe).
  mlir::LLVM::GlobalOp moduleGlobalStorage(mlir::Operation *op,
                                           llvm::StringRef name);
  mlir::LogicalResult lowerStaticCtypesGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index);
  mlir::LogicalResult
  lowerStaticCtypesLibraryGetItem(py::GetItemOp op,
                                  const RuntimeBundle &container);
  mlir::LogicalResult lowerStaticCtypesModuleCall(py::CallOp op,
                                                  const RuntimeBundle &receiver,
                                                  llvm::StringRef methodName);
  mlir::LogicalResult
  lowerStaticCtypesTypeObjectCall(py::CallOp op, const RuntimeBundle &callable);
  mlir::LogicalResult
  lowerCtypesCallbackConstruction(py::CallOp op, const RuntimeBundle &callable,
                                  llvm::ArrayRef<const RuntimeBundle *> sources);
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
  mlir::LogicalResult
  lowerStaticCtypesValueAttrSet(py::AttrSetOp op, const RuntimeBundle &object,
                                const RuntimeBundle *value);
  mlir::LogicalResult lowerStaticCtypesCall(py::CallOp op,
                                            const RuntimeBundle &callable);
  mlir::LogicalResult
  lowerStaticCtypesNativeCall(py::CallOp op, const RuntimeBundle &callable);
  mlir::FailureOr<RuntimeValue>
  materializePrimitiveI64Object(mlir::Operation *op,
                                const RuntimeBundle &bundle);
  mlir::FailureOr<RuntimeValue>
  materializePrimitiveI64ObjectAtCurrentInsertion(mlir::Operation *op,
                                                  const RuntimeBundle &bundle);
  mlir::FailureOr<RuntimeValue>
  materializeObjectEvidenceValue(mlir::Operation *op,
                                 const RuntimeBundle &bundle,
                                 llvm::StringRef purpose);
  mlir::FailureOr<RuntimeBundle> materializeObjectBundleForStorage(
      mlir::Operation *op, const RuntimeBundle &bundle,
      mlir::Type storageContract, llvm::StringRef purpose);
  bool objectShapeMatches(llvm::StringRef contract,
                          mlir::ValueRange values) const;
  bool isBuiltinsObjectHandleType(mlir::Type type) const;
  bool isErasedObjectStorageType(mlir::Type type) const;
  bool isBuiltinsObjectContract(mlir::Type type) const;
  const RuntimeBundle *
  concreteObjectForOwnership(const RuntimeBundle &bundle) const;
  mlir::FailureOr<RuntimeBundle> boxRuntimeObject(mlir::Operation *op,
                                                  const RuntimeBundle &source,
                                                  bool retainPayload);
  mlir::FailureOr<RuntimeBundle> boxRuntimeObjectAtCurrentInsertion(
      mlir::Operation *op, const RuntimeBundle &source, bool retainPayload);
  mlir::FailureOr<mlir::Value> objectPhysicalHeader(mlir::Operation *op,
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
  mlir::FailureOr<RuntimeValue>
  materializeNonOwningDeadObjectValue(mlir::Operation *op, mlir::Type contract,
                                      llvm::StringRef purpose);
  mlir::FailureOr<RuntimeValue>
  materializeDeadObjectValueImpl(mlir::Operation *op, mlir::Type contract,
                                 llvm::StringRef purpose,
                                 DeadObjectStorage storage);
  mlir::FailureOr<RuntimeValue>
  materializeClassObjectValue(mlir::Operation *op, py::ClassOp classOp,
                              mlir::Type contract, llvm::StringRef purpose);
  mlir::LogicalResult materializeStringObject(mlir::Operation *op,
                                              llvm::StringRef text,
                                              RuntimeBundle &bundle);
  mlir::LogicalResult materializeBytesObject(mlir::Operation *op,
                                             llvm::StringRef data,
                                             RuntimeBundle &bundle);
  bool needsDefaultObjectRepr(const RuntimeBundle &object) const;
  mlir::LogicalResult materializeDefaultObjectRepr(mlir::Operation *op,
                                                   const RuntimeBundle &object,
                                                   RuntimeBundle &bundle);
  // Statically-known source-class receivers dispatch `__repr__` as a direct
  // call to the compiled method (the erased-element counterpart is the boxed
  // repr hook). Returns false when the receiver has no source-class __repr__.
  mlir::FailureOr<bool> emitSourceClassReprCall(mlir::Operation *op,
                                                const RuntimeBundle &object,
                                                RuntimeBundle &result);
  // Erased (`builtins.object`) receivers dispatch `__repr__` through the boxed
  // repr hook on their class id, trapping when no conforming __repr__ exists.
  mlir::LogicalResult emitBoxedReprHookCall(mlir::Operation *op,
                                            const RuntimeBundle &object,
                                            RuntimeBundle &result);
  mlir::func::FuncOp findRetainFunction() const;
  mlir::LogicalResult retainAggregateSlot(mlir::Operation *op,
                                          mlir::Type slotType,
                                          mlir::ValueRange values,
                                          llvm::StringRef slotName);
  mlir::LogicalResult retainAggregateSlot(mlir::Operation *op,
                                          const RuntimeBundle &slotValue,
                                          llvm::StringRef slotName);
  mlir::LogicalResult releaseAggregateSlot(mlir::Operation *op,
                                           mlir::Type slotType,
                                           mlir::ValueRange values,
                                           llvm::StringRef slotName);
  mlir::LogicalResult releaseAggregateSlot(mlir::Operation *op,
                                           const RuntimeBundle &slotValue,
                                           llvm::StringRef slotName);
  mlir::LogicalResult
  replaceAggregateSlot(mlir::Operation *op, mlir::Type oldType,
                       mlir::ValueRange oldValues, mlir::Type newType,
                       mlir::ValueRange newValues, llvm::StringRef slotName);
  mlir::LogicalResult replaceAggregateSlot(
      mlir::Operation *op, mlir::Type oldType, mlir::ValueRange oldValues,
      const RuntimeBundle *oldSlotValue, mlir::Type newType,
      const RuntimeBundle &newSlotValue, llvm::StringRef slotName,
      bool releaseMissingOldObjectSlot = true);
  mlir::LogicalResult retainAggregateSlot(mlir::Operation *op,
                                          mlir::Type slotType,
                                          mlir::ValueRange values,
                                          llvm::StringRef slotName,
                                          unsigned depth);
  mlir::LogicalResult releaseAggregateSlot(
      mlir::Operation *op, mlir::Type slotType, mlir::ValueRange values,
      llvm::StringRef slotName,
      llvm::ArrayRef<ownership::RuntimeDeallocator> deallocators,
      unsigned depth);
  std::uint64_t collectionInitialCapacity(std::uint64_t arity) const;
  mlir::FailureOr<RuntimeBundle>
  materializePayloadObjectBundle(mlir::Operation *op,
                                 const RuntimeBundle &value);
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
  objectPayloadHandleWords(mlir::Operation *op, const RuntimeBundle &value,
                           bool ownsPayload = true);
  mlir::LogicalResult initializeSequencePayload(
      mlir::Operation *op, RuntimeBundle &container,
      llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> elements);
  mlir::LogicalResult ensureSequencePayloadCapacity(mlir::Operation *op,
                                                    RuntimeBundle &container,
                                                    unsigned index,
                                                    llvm::StringRef label);
  mlir::LogicalResult storeSequencePayloadElement(mlir::Operation *op,
                                                  RuntimeBundle &container,
                                                  unsigned index,
                                                  const RuntimeBundle &element);
  // Runtime-index element store for runtime-mode sequences; the caller is
  // responsible for having grown the payload via the ensure_capacity
  // primitive.
  mlir::LogicalResult storeSequencePayloadElementAt(
      mlir::Operation *op, RuntimeBundle &container, mlir::Value logicalIndex,
      const RuntimeBundle &element);
  mlir::LogicalResult clearSequencePayloadElement(mlir::Operation *op,
                                                  RuntimeBundle &container,
                                                  unsigned index);
  mlir::LogicalResult
  initializeDictPayload(mlir::Operation *op, RuntimeBundle &container,
                        llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> keys,
                        llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> values);
  mlir::LogicalResult ensureDictPayloadCapacity(mlir::Operation *op,
                                                RuntimeBundle &container,
                                                unsigned index);
  mlir::LogicalResult storeDictKeyPayload(mlir::Operation *op,
                                          RuntimeBundle &container,
                                          unsigned index,
                                          const RuntimeBundle &key);
  mlir::LogicalResult storeDictValuePayload(mlir::Operation *op,
                                            RuntimeBundle &container,
                                            unsigned index,
                                            const RuntimeBundle &value);
  mlir::LogicalResult clearDictKeyPayload(mlir::Operation *op,
                                          RuntimeBundle &container,
                                          unsigned index);
  mlir::LogicalResult clearDictValuePayload(mlir::Operation *op,
                                            RuntimeBundle &container,
                                            unsigned index);
  mlir::LogicalResult clearDictPayloadEntry(mlir::Operation *op,
                                            RuntimeBundle &container,
                                            unsigned index);
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
  // Retain an evidence-selected container element through its contract's `own`
  // primitive (inserted right after the element's defining ops, where it is
  // provably alive) so it survives the container's release. Returns the
  // retained element, or nullopt when the contract has no usable `own`
  // primitive (callers fall back to the borrowed binding). With `atOperation`
  // the retain is placed at `op` instead of after the element's defs — for
  // inline-constructed locals that are uninitialized at their defs; the caller
  // must pin the container's liveness past the retain.
  std::optional<RuntimeValue> retainEvidenceElement(mlir::Operation *op,
                                                    const RuntimeValue &value,
                                                    bool atOperation = false);
  mlir::LogicalResult pinContainerLiveness(mlir::Operation *op,
                                           const RuntimeBundle &container);
  mlir::FailureOr<std::optional<RuntimeValue>>
  retainEvidenceElementWithFallback(mlir::Operation *op,
                                    const RuntimeValue &value,
                                    const RuntimeBundle *container);
  // Contracts with a manifest `box` primitive live in container slots in the
  // primitive's RESULT shape (e.g. bool: an immortal singleton header) and
  // `unbox` back to their canonical value group on load — value semantics, so
  // unboxed elements need no retain.
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
  slotStorageShapesFor(mlir::Operation *op, mlir::Type contract,
                       llvm::StringRef purpose);
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
  unboxSlotElementValues(mlir::Operation *op, mlir::Type contract,
                         llvm::ArrayRef<mlir::Value> values);
  mlir::LogicalResult
  bindRetainedEvidenceValue(mlir::Operation *op, mlir::Value resultValue,
                            llvm::StringRef label, const RuntimeValue &value,
                            const RuntimeBundle *container = nullptr);
  mlir::LogicalResult
  bindRetainedEvidenceBundle(mlir::Operation *op, mlir::Value resultValue,
                             RuntimeBundle bundle,
                             const RuntimeBundle *container = nullptr);
  mlir::FailureOr<RuntimeBundle>
  selectEvidenceObjectByMatch(mlir::Operation *op, mlir::Value resultValue,
                              llvm::ArrayRef<RuntimeValue> candidates,
                              mlir::ValueRange matches, llvm::StringRef label,
                              llvm::StringRef missingContract,
                              llvm::StringRef missingMessage,
                              bool raiseOnMiss = true);
  // Both take the iterator bundle by value: they bind results into
  // `valueBundles` (a DenseMap) mid-lowering, which invalidates references
  // into the map.
  // Rank-1 memref view over a boxed pointer/size word pair (inline
  // descriptor assembly; borrow-only).
  static mlir::Value memrefFromBoxWords(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value pointerWord,
                                        mlir::Value sizeWord,
                                        mlir::MemRefType type);
  // Per-program release hook: dispatches a boxed slot's class id to the
  // matching manifest deallocator (the single release implementation).
  mlir::LogicalResult generateBoxedReleaseHook();
  // Uniform boxed-object method dispatch. Builds a per-program hook
  // `(ptr box, i64 class_id) -> (calleeResults..., i1 handled)` that dispatches
  // the class id to the matching manifest function GENERICALLY — no per-type
  // special-casing — reconstructing its memref arguments from the shared box
  // word layout (slot words (4+i, 9+i) = physical value i, so a compiled
  // source-class method taking (self box, field views...) conforms as-is).
  // `selects` picks which manifest functions participate (e.g. a deallocator
  // attribute, or a `__repr__` method); every selected function must share
  // `calleeResultTypes`. When `sourceClassMethodName` is non-empty, compiled
  // source-class methods of that name join the dispatch through the same
  // conformance checks.
  mlir::LogicalResult generateBoxedMethodHook(
      llvm::StringRef hookName,
      llvm::function_ref<bool(mlir::func::FuncOp)> selects,
      mlir::TypeRange calleeResultTypes, bool shareExceptionSubclasses,
      llvm::StringRef sourceClassMethodName = "");
  // repr instance of the uniform dispatch: class id -> the manifest `__repr__`
  // returning a `builtins.str`, for container __repr__ over erased elements.
  mlir::LogicalResult generateBoxedReprHook();
  mlir::LogicalResult lowerListEvidenceNext(py::NextOp op,
                                            RuntimeBundle iterator);
  // Loop-body generator state-machine transform (GeneratorStateMachine.cpp).
  struct GeneratorResumeInfo {
    std::string cloneName;
    unsigned frameWidth = 0;
    unsigned argumentCount = 0;
  };
  llvm::StringMap<GeneratorResumeInfo> generatorResumeClones;
  mlir::LogicalResult buildGeneratorResumeCloneSignatures();
  mlir::LogicalResult buildGeneratorResumeBodies();
  mlir::FailureOr<SourceGeneratorResumeResult>
  emitStateMachineGeneratorResume(mlir::Operation *op,
                                  const RuntimeBundle &iterator,
                                  const GeneratorResumeInfo &info,
                                  bool useCurrentInsertionPoint = false);
  mlir::LogicalResult lowerListRuntimeNext(py::NextOp op,
                                           RuntimeBundle iterator);
  mlir::FailureOr<bool> lowerRuntimeSequenceGetItem(py::GetItemOp op,
                                                    const RuntimeBundle &container,
                                                    const RuntimeBundle &index);
  mlir::FailureOr<bool> lowerRuntimeDictGetItem(py::GetItemOp op,
                                                const RuntimeBundle &container,
                                                const RuntimeBundle &index);
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
  mlir::LogicalResult lowerBytesConstant(py::BytesConstantOp op);
  mlir::LogicalResult lowerIntConstant(py::IntConstantOp op);
  mlir::LogicalResult lowerFloatConstant(py::FloatConstantOp op);
  mlir::LogicalResult lowerBoolConstant(py::BoolConstantOp op);
  mlir::LogicalResult lowerNone(py::NoneOp op);
  mlir::LogicalResult lowerCastFromPrim(py::CastFromPrimOp op);
  mlir::LogicalResult lowerUnionWrap(py::UnionWrapOp op);
  mlir::LogicalResult lowerUnionTest(py::UnionTestOp op);
  mlir::LogicalResult lowerUnionUnwrap(py::UnionUnwrapOp op);
  mlir::LogicalResult lowerClassTest(py::ClassTestOp op);
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
  mlir::LogicalResult
  bindRuntimeCallBundle(mlir::Operation *op, mlir::Type resultType,
                        const EmittedRuntimeCall &emitted,
                        const RuntimeBundle *receiverEvidence,
                        RuntimeBundle &result);
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
  mlir::LogicalResult emitRaiseExceptionBundle(mlir::Operation *op,
                                               const RuntimeBundle &exception,
                                               bool discardCurrentException);
  mlir::LogicalResult lowerRaise(py::RaiseOp op);
  mlir::LogicalResult lowerRaiseCurrent(py::RaiseCurrentOp op);
  mlir::LogicalResult lowerExceptMatch(py::ExceptMatchOp op);
  mlir::LogicalResult lowerExceptCurrentMatch(py::ExceptCurrentMatchOp op);
  mlir::LogicalResult lowerExceptCurrentValue(py::ExceptCurrentValueOp op);
  mlir::LogicalResult emitTracebackFrame(mlir::Operation *op);
  mlir::LogicalResult lowerCall(py::CallOp op);
  mlir::LogicalResult lowerBoundMethodCall(py::CallOp op,
                                           const RuntimeBundle &receiver,
                                           llvm::StringRef methodName);
  mlir::LogicalResult lowerFutureResultEvidence(mlir::Operation *op,
                                                mlir::Value resultValue,
                                                const RuntimeBundle &receiver,
                                                llvm::StringRef label);
  mlir::LogicalResult bundleCoroutineBodyResults(mlir::Operation *op,
                                                 mlir::Value resultValue,
                                                 mlir::ValueRange values,
                                                 RuntimeBundle &result);
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
  mlir::LogicalResult emitAsyncFunctionTargetCallResult(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      llvm::ArrayRef<const RuntimeBundle *> sources, RuntimeBundle &result);
  mlir::LogicalResult emitAsyncFunctionTargetCallResult(
      mlir::Operation *op, mlir::Value resultValue, mlir::func::FuncOp target,
      llvm::StringRef targetName, llvm::ArrayRef<const RuntimeBundle *> sources,
      RuntimeBundle &result);
  mlir::LogicalResult lowerGeneratorFunctionTargetCall(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult emitGeneratorFunctionTargetCallResult(
      mlir::Operation *op, mlir::Value resultValue, mlir::func::FuncOp target,
      llvm::StringRef targetName, llvm::ArrayRef<const RuntimeBundle *> sources,
      RuntimeBundle &result);
  mlir::LogicalResult emitSourceFunctionTargetCallResult(
      mlir::Operation *op, mlir::Type expectedResult, mlir::func::FuncOp target,
      llvm::StringRef targetName, llvm::ArrayRef<const RuntimeBundle *> sources,
      RuntimeBundle &result);
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
  mlir::FailureOr<SourceGeneratorResumeResult>
  emitSourceGeneratorResumeDispatch(mlir::Operation *op, mlir::Type elementType,
                                    const RuntimeBundle &iterator,
                                    bool useCurrentInsertionPoint = false,
                                    std::optional<RuntimePrimitiveI64Evidence>
                                        sentI64Evidence = std::nullopt);
  mlir::LogicalResult lowerSourceGeneratorNext(py::NextOp op,
                                               const RuntimeBundle &iterator);
  mlir::LogicalResult
  lowerSourceGeneratorSend(py::CallOp op, const RuntimeBundle &receiver,
                           llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult
  lowerSourceGeneratorThrow(py::CallOp op, const RuntimeBundle &receiver,
                            llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult lowerEnter(py::EnterOp op);
  mlir::LogicalResult lowerExit(py::ExitOp op);
  mlir::LogicalResult lowerAEnter(py::AEnterOp op);
  mlir::LogicalResult lowerAExit(py::AExitOp op);
  mlir::LogicalResult lowerAIter(py::AIterOp op);
  mlir::LogicalResult lowerANext(py::ANextOp op);
  mlir::LogicalResult lowerAwait(py::AwaitOp op);
  mlir::LogicalResult lowerCoroutineObjectAwait(mlir::Operation *op,
                                                mlir::Value resultValue,
                                                RuntimeBundle &awaitable,
                                                llvm::StringRef label);
  mlir::LogicalResult lowerCoroutineStorageTargetIdAwait(
      mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &awaitable,
      llvm::StringRef label);
  mlir::LogicalResult lowerAwaitIteratorResult(mlir::Operation *op,
                                               mlir::Value resultValue,
                                               RuntimeBundle &iterator,
                                               llvm::StringRef label);
  mlir::LogicalResult lowerGeneralAwaitableIterator(py::AwaitOp op,
                                                    RuntimeBundle &awaitable);
  mlir::LogicalResult lowerRound(py::RoundOp op);
  mlir::LogicalResult lowerIncRef(py::IncRefOp op);
  mlir::LogicalResult lowerDecRef(py::DecRefOp op);
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
                               const RuntimeBundle *&argument);
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
  mlir::LogicalResult lowerRuntimeValueSelect(mlir::arith::SelectOp select);
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
  mlir::LogicalResult eraseSourceGeneratorBodyFunctions();
  mlir::LogicalResult synthesizeSourceClassDeallocators();
  mlir::LogicalResult eraseCallableLogicalEntryArgs();
  mlir::LogicalResult dropUnusedLogicalBlockArguments();
  mlir::LogicalResult eraseLoweredPyOps();

  mlir::ModuleOp module;
  unsigned callbackThunkCounter = 0;
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  RuntimeManifestIndex manifest;
  llvm::DenseMap<mlir::Value, RuntimeBundle> valueBundles;
  llvm::DenseMap<mlir::Value, mlir::Operation *> ownedLocalObjectMarkers;
  llvm::StringMap<ReturnedValueSummary> returnedValueSummaries;
  llvm::StringMap<ReturnedCallableSummary> returnedCallableSummaries;
  llvm::StringMap<ReturnedCoroutineSummary> returnedCoroutineSummaries;
  llvm::StringMap<ReturnedObjectEvidenceSummary>
      returnedObjectEvidenceSummaries;
  llvm::StringMap<ReturnedStaticObjectSummary> returnedStaticObjectSummaries;
  llvm::StringMap<llvm::SmallVector<mlir::Type, 8>>
      callableProtocolArgumentABIs;
  llvm::StringMap<llvm::SmallVector<CallableProtocolSpecialization, 4>>
      callableProtocolSpecializations;
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

} // namespace py::lowering
