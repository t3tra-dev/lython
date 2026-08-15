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
#include "llvm/ADT/SetVector.h"
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
    // Object-family yield lanes: the full physical span of the yielded value
    // (concrete contract parts, plus the trailing (i64, i1) evidence pair for
    // builtins.int). Empty for the legacy pure-pair int tier, where `value` /
    // `valid` alone carry the yield.
    llvm::SmallVector<mlir::Value, 6> lanePhysicals;
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
  // Does any class in this module name `contract` as a base? A `type[X]` has
  // an empty physical shape only because its type decides the class, and that
  // stops being true the moment X is subclassed. Built once and cached: the
  // question is asked per ABI lane and the answer is a property of the module.
  bool contractIsSubclassed(llvm::StringRef contract) const;
  // Nearest manifest exception ancestor (a contract with a `raise`
  // primitive) along a source class's emitter-computed linearization
  // (`mro_names`); nullopt for non-exception source classes. Instances of
  // exception-backed classes use the ancestor's physical representation --
  // the runtime exception object with the source class's id in its header.
  std::optional<std::string>
  exceptionAncestorContract(py::ClassOp classOp) const;
  std::optional<std::string> exceptionAncestorContractFor(mlir::Type type) const;
  // Class id of the next exception class after `classOp` in its MRO (a user
  // exception base's source id, else the builtin ancestor's manifest id).
  std::optional<std::int64_t>
  userExceptionParentClassId(py::ClassOp classOp) const;
  // Per-program hooks the native support module links against:
  // __ly_user_exception_base_class_id (id -> parent id, 0 unknown) and
  // __ly_user_exception_class_name (id -> C-string ptr, null unknown).
  mlir::LogicalResult synthesizeUserExceptionHooks();
  // Class id of an except-clause handler type (manifest or source class).
  mlir::FailureOr<std::int64_t> handlerClassId(mlir::Operation *op,
                                               mlir::Type handler) const;
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
  // EVERY object-contract field is stored BOX-FRONTED: one box16 slot whose
  // pointer is fixed for the instance's lifetime, with the field's value
  // reached through its words. A store is then a store to a heap slot rather
  // than a replacement of the instance's SSA lanes, which is what makes it
  // observable through a function boundary and across a branch. The exceptions
  // are the contracts that need no indirection: int/bool live in an instance
  // header word, a zero-lane contract has nothing to hold, and a union is not
  // one object.
  bool classFieldStoredBoxed(mlir::Type fieldContract) const;
  // The storage a field occupies in the instance's expansion, by POSITION: a
  // header-word field (int/bool) takes none, a box-fronted field one box16, and
  // the residual shapes their contract's own lanes.
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
  classFieldStorageValueTypes(mlir::Operation *op, mlir::Type fieldContract,
                              unsigned fieldIndex,
                              llvm::StringRef purpose) const;
  // Stores a header-word field's value into the instance header. Shared by
  // attr.set and the no-__init__ field-record initializer, which used to write
  // the placeholder lanes instead and so read back zero.
  mlir::LogicalResult storePrimitiveFieldSlot(mlir::Operation *op,
                                              const RuntimeBundle &object,
                                              const RuntimeBundle &value,
                                              mlir::Type fieldType,
                                              unsigned fieldIndex,
                                              llvm::StringRef fieldName);
  mlir::FailureOr<RuntimeBundle>
  storeBoxedFieldPayloadInPlace(mlir::Operation *op, mlir::Value box,
                                const RuntimeBundle &value,
                                llvm::StringRef slotName);
  mlir::LogicalResult updateBoxedFieldPayloadWords(mlir::Operation *op,
                                                   mlir::Value box,
                                                   const RuntimeBundle &payload,
                                                   llvm::StringRef slotName);
  // R6 nonlocal cells: emitter-synthesized one-field classes whose field
  // slot is a box16 written IN PLACE (never spliced into new SSA lanes), so
  // every frame sharing the cell instance observes one mutable slot.
  static bool isCellClassOp(py::ClassOp classOp);
  mlir::LogicalResult lowerCellAttrGet(py::AttrGetOp op,
                                       const RuntimeBundle &object,
                                       py::ClassOp classOp,
                                       unsigned fieldIndex);
  // R2 user exception fields: an exception-backed class has NO field lanes
  // (its ABI is the taxonomy's 3-word header + message), so its declared
  // fields live in a [count, count x box16] block hung off extended header
  // word 4 — reached only through the BaseException payload primitives.
  mlir::FailureOr<mlir::Value>
  exceptionFieldBlockWord(mlir::Operation *op, const RuntimeBundle &object,
                          py::ClassOp classOp);
  mlir::LogicalResult lowerExceptionFieldAttrGet(py::AttrGetOp op,
                                                 const RuntimeBundle &object,
                                                 py::ClassOp classOp,
                                                 unsigned fieldIndex);
  mlir::LogicalResult lowerExceptionFieldAttrSet(py::AttrSetOp op,
                                                 const RuntimeBundle &object,
                                                 const RuntimeBundle &value,
                                                 py::ClassOp classOp,
                                                 unsigned fieldIndex);
  mlir::LogicalResult lowerCellAttrSet(py::AttrSetOp op,
                                       const RuntimeBundle &object,
                                       const RuntimeBundle &value,
                                       py::ClassOp classOp,
                                       unsigned fieldIndex);
  mlir::LogicalResult writeBackFieldAlias(mlir::Operation *op,
                                          const RuntimeBundle &updatedField);
  mlir::LogicalResult rebindMutatedContainer(mlir::Operation *op,
                                             const RuntimeBundle &receiver,
                                             mlir::ValueRange values,
                                             RuntimeBundle &rebound);
  mlir::LogicalResult
  promoteInteriorViewForTransfer(mlir::Operation *op,
                                 const RuntimeBundle &receiver,
                                 llvm::StringRef slotName,
                                 mlir::func::FuncOp mutation);
  std::optional<unsigned> findUnionMemberIndex(py::UnionType unionType,
                                               mlir::Type member) const;
  mlir::FailureOr<unsigned>
  requireUnionMemberIndex(mlir::Operation *op, py::UnionType unionType,
                          mlir::Type member, llvm::StringRef purpose) const;
  mlir::FailureOr<unsigned>
  unionMemberValueOffset(mlir::Operation *op, py::UnionType unionType,
                         unsigned memberIndex, llvm::StringRef purpose) const;
  // `lanesSource`, when given, receives the bundle whose physical values
  // actually went into `values` -- which is not `source` when a lazily-boxed
  // int had to be materialized to fill the member lanes. A caller that
  // remembers the active member must remember THAT one; see lowerUnionWrap.
  mlir::LogicalResult
  appendUnionRuntimeValues(mlir::Operation *op, py::UnionType resultUnion,
                           const RuntimeBundle &source, mlir::Type sourceType,
                           llvm::SmallVectorImpl<mlir::Value> &values,
                           RuntimeBundle *lanesSource = nullptr);
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
  // Records that the clone containing `op` just made a decision from a raw
  // lane whose validity bit was not statically true, so every `valid` it
  // returns from here on must be forced to false.
  void poisonPrimitiveI64CloneSpeculation(mlir::Operation *op,
                                          mlir::Value stillValid);
  // Null unless the enclosing clone was poisoned somewhere; otherwise an i1
  // that is true only if the raw lane is still trustworthy.
  mlir::Value primitiveI64CloneSpeculationIntact(mlir::Operation *op,
                                                 mlir::func::FuncOp clone);
  mlir::LogicalResult foldUnprovenPrimitiveI64Speculations();
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
  // The two NATIVE-cell paths (see nativeGlobalCell). An ADDRESS global is a
  // machine word with process lifetime, not a Python object; a
  // runtime-internal module's `int` global keeps the word so a signal handler
  // may read it. Everything else takes the object path above.
  mlir::LogicalResult lowerAddressGlobalGet(py::GlobalGetOp op);
  mlir::LogicalResult lowerAddressGlobalSet(py::GlobalSetOp op,
                                            const RuntimeBundle &value);
  mlir::LogicalResult lowerNativeIntGlobalGet(py::GlobalGetOp op);
  mlir::LogicalResult lowerNativeIntGlobalSet(py::GlobalSetOp op,
                                              const RuntimeBundle &value);
  static bool isAddressGlobalType(mlir::Type type);
  // `cellType` is what the cell holds: `i64` for the bound flag, sizes and
  // scalars, `!llvm.ptr` for the `_p<i>` slots. A pointer slot holding a
  // pointer is the whole reason this takes a type -- see lowerObjectGlobalSet.
  mlir::LLVM::GlobalOp moduleObjectGlobalCell(mlir::Operation *op,
                                              llvm::StringRef name,
                                              llvm::StringRef suffix,
                                              mlir::Type cellType);
  mlir::Value loadObjectGlobalWord(mlir::Operation *op, llvm::StringRef name,
                                   llvm::StringRef suffix);
  void storeObjectGlobalWord(mlir::Operation *op, llvm::StringRef name,
                             llvm::StringRef suffix, mlir::Value word);
  mlir::Value loadObjectGlobalPointer(mlir::Operation *op,
                                      llvm::StringRef name,
                                      llvm::StringRef suffix);
  void storeObjectGlobalPointer(mlir::Operation *op, llvm::StringRef name,
                                llvm::StringRef suffix, mlir::Value pointer);
  mlir::LogicalResult
  loadObjectGlobalValues(mlir::Operation *op, llvm::StringRef name,
                         llvm::ArrayRef<mlir::Type> valueTypes,
                         llvm::SmallVectorImpl<mlir::Value> &values);
  mlir::LogicalResult lowerGlobalSet(py::GlobalSetOp op);
  // Process-lifetime i64 storage for a module-level int global, created on
  // first use. Reads/writes are plain load/store (async-signal-safe).
  mlir::LLVM::GlobalOp nativeGlobalCell(mlir::Operation *op,
                                        llvm::StringRef name);
  mlir::Value loadNativeGlobalWord(mlir::Operation *op, llvm::StringRef name);
  void storeNativeGlobalWord(mlir::Operation *op, llvm::StringRef name,
                             mlir::Value word);
  mlir::LogicalResult lowerStaticCtypesGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index);
  mlir::LogicalResult
  lowerStaticCtypesLibraryGetItem(py::GetItemOp op,
                                  const RuntimeBundle &container);
  // The one body behind both spellings of a library symbol access,
  // `lib["write"]` and `lib.write`: they differ only in where the name and
  // the alias owner come from.
  mlir::LogicalResult
  bindStaticCtypesLibrarySymbol(mlir::Operation *op,
                                const RuntimeBundle &library,
                                llvm::StringRef symbol, mlir::Value aliasOwner,
                                mlir::Value result);
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
  // The builtins.object methods a source class may INHERIT and run against its
  // own instance (CPython's object.__eq__/__ne__/__hash__ — identity-defined,
  // so object's implementation is correct for any subclass). object's
  // __repr__/__str__ are deliberately absent: they render "<object object at
  // ...>" and would lose the class name materializeDefaultObjectRepr keeps.
  static bool isInheritedObjectDunder(llvm::StringRef methodName);
  // True when this call is such an inherited dunder reached with a source-class
  // receiver. The receiver has to be BOXED for it: the instance's own leading
  // physical has the same memref type as a builtins.object handle but not the
  // same meaning — only a payload box carries the entity word the callee reads
  // as the object's identity — so aliasing the storage would make every
  // instance of a class compare equal.
  bool usesInheritedObjectDunder(const RuntimeSymbol &symbol,
                                 const RuntimeBundle &source) const;
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
      bool releaseMissingOldObjectSlot = true, bool releaseOldSlot = true);
  // Does the release machinery see the CURRENT expansion of `logicalValue`,
  // i.e. will a field re-root be republished to it? Only an instance
  // constructed in this frame carries the owned-local marker that publishes
  // it; for anything else the release still names the birth expansion, so
  // releasing the replaced value at the store would release it a second time.
  bool ownedLocalObjectMarkerFollowsExpansion(mlir::Value logicalValue) const;
  // Is the value being stored into a slot the SAME entity the slot already
  // holds, reached through mutation primitives that consume it and hand it
  // back? `ks = self._kids; ks.append(v); self._kids = ks` is spelled that way
  // by the current ABI, so the store is a SELF-store: the token the slot holds
  // never left it, and releasing the pre-mutation lanes there would hand the
  // deallocator storage the primitive has already reallocated.
  bool aggregateSlotStoreIsSelfStore(mlir::ValueRange oldValues,
                                     mlir::ValueRange newValues) const;
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
  static bool isMutableContainerContractName(llvm::StringRef contract);
  // The four contracts whose payload is an array of 16-word element boxes.
  // Named for the boundaries that are polymorphic over all of them and so can
  // name none of their shapes (`join`, `frozenset(iterable)`).
  static bool isSequenceLikeContractName(llvm::StringRef contract);
  static void demoteMutableContainerEvidence(RuntimeBundle &bundle);
  void dropObjectFieldEvidence(RuntimeBundle &bundle);
  void demoteMutableContainerEvidenceFor(mlir::Value value);
  void demoteMutableContainerArgumentEvidence(py::CallOp op);
  mlir::FailureOr<mlir::Value>
  rawSequenceIndexValue(mlir::Operation *op, mlir::Value indexValue,
                        const RuntimeBundle &index);
  mlir::FailureOr<RuntimeBundle>
  materializePayloadObjectBundle(mlir::Operation *op,
                                 const RuntimeBundle &value);
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
  objectPayloadHandleWords(mlir::Operation *op, const RuntimeBundle &value,
                           bool ownsPayload = true);
  // True when `op` is the ONLY user of `value` — i.e. the value is a
  // temporary this op consumes, not a binding that outlives it. Container
  // literals use it to decide whether storing an element may take over the
  // source's reference (CollectionPayload.cpp).
  static bool valueIsConsumedOnlyBy(mlir::Value value, mlir::Operation *op);
  // `logicalSources`, when non-empty, is parallel to `elements` and names the
  // SSA value each element came from; a null entry means "freshly minted
  // temporary". An empty list means every owned element is treated as a
  // temporary (the pre-threading behaviour).
  mlir::LogicalResult initializeSequencePayload(
      mlir::Operation *op, RuntimeBundle &container,
      llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> elements,
      llvm::ArrayRef<mlir::Value> logicalSources = {});
  // Which piece of a container's interior state a view names. `Meta` is the
  // {length, capacity} pair; `Primary` is `items` for a sequence and `keys`
  // for a mapping; `Secondary` is a mapping's `values`; `Present` is the
  // per-slot occupancy flags.
  enum class ContainerInterior { Primary, Secondary, Present };
  // Borrowed rank-1 i64 view of one piece of a container's interior state,
  // derived AT THE POINT OF USE.
  //
  // Why an accessor and not a lane subscript: for a contract whose interior
  // state lives behind the entity handle there is no lane to subscript, and
  // for one that still carries lanes the subscript is the lane. Both spellings
  // answer the same question, and the accessor is the only place that has to
  // know which one a contract uses -- so a contract converts by changing its
  // `ly.runtime.shape` and its manifest bodies, not by editing every reader.
  //
  // Why callers must not cache the result: a handle-fronted view is only valid
  // while the handle's provenance is live and its base word unchanged. Naming
  // it across a reallocation is exactly the staleness the one-lane form
  // removes (rfc/memory-safety-proof.md, `Interior`).
  mlir::FailureOr<mlir::Value>
  containerInteriorView(mlir::Operation *op, const RuntimeBundle &container,
                        ContainerInterior which, llvm::StringRef label);
  // True when `container`'s interior state is reached through the entity
  // handle rather than through payload lanes beside it.
  bool containerIsHandleFronted(const RuntimeBundle &container) const;
  // The gate for every runtime-mode container path: does this container carry
  // a payload the lowering can reach? Never spell the lane count at a use
  // site -- a count that stops matching reads as "no payload here" and takes
  // the evidence path instead of failing.
  bool containerHasRuntimePayload(const RuntimeBundle &container) const;
  // (memref, index) pair naming one slot of a container's {length, capacity}
  // pair. Not a view: reading the pair out of the entity's own storage keeps
  // the handle an operand of the load, so nothing has to pin it.
  mlir::FailureOr<std::pair<mlir::Value, mlir::Value>>
  containerMetaSlot(mlir::Operation *op, const RuntimeBundle &container,
                    std::int64_t slot, llvm::StringRef label);
  mlir::FailureOr<mlir::Value> loadContainerLength(
      mlir::Operation *op, const RuntimeBundle &container,
      llvm::StringRef label);
  mlir::LogicalResult storeContainerLength(mlir::Operation *op,
                                          const RuntimeBundle &container,
                                          mlir::Value length,
                                          llvm::StringRef label);
  mlir::LogicalResult adjustContainerLength(mlir::Operation *op,
                                            const RuntimeBundle &container,
                                            std::int64_t delta,
                                            llvm::StringRef label);
  // Reading the length is how an evidence-backed collection records that the
  // runtime payload was consulted; the value itself is discarded.
  mlir::LogicalResult touchContainerEvidenceUse(mlir::Operation *op,
                                                const RuntimeBundle &container,
                                                llvm::StringRef label);
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
                        llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> values,
                        llvm::ArrayRef<mlir::Value> logicalKeySources = {},
                        llvm::ArrayRef<mlir::Value> logicalValueSources = {});
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
  // ⭐ The bookkeeping half of the above, without the retain: mark the element
  // as a frame-owned local so the ordinary owned-result machinery RELEASES it,
  // and take no reference because the element already arrives with one.
  //
  // The two halves were one function, and that coupling was a measured leak.
  // `LyObject_FromSlot` (runtime/modules/builtins.mlir) allocates a fresh box
  // and stores 1 into its refcount word -- it is declared
  // `ly.ownership.owned_results = [0]` and it means it. Retaining that result
  // as well left the counter at 2 against one release, so every boxed slot read
  // leaked one object, unbounded (measured: 10 reads -> 10 roots, 50 -> 50).
  //
  // Why NOT decide it from `RuntimeValue::ownership`: `bindEvidenceObjectResult`
  // hardcodes `ownsObject=false` and discards that field, so setting it here
  // would change nothing while reading as though it did. And the retain path is
  // shared with three sites that inherit ownership from a bundle
  // (GetItemOps.cpp bindRetainedEvidenceBundle and the two selection merges),
  // where an `Own` inherited from elsewhere would then silently drop a retain
  // that IS required. The caller knows which primitive produced the element;
  // the shared helper does not.
  std::optional<RuntimeValue> rootOwnedEvidenceElement(mlir::Operation *op,
                                                       const RuntimeValue &value,
                                                       bool atOperation = false);
  mlir::LogicalResult forEachActiveUnionMember(
      mlir::Operation *op, py::UnionType unionType, mlir::ValueRange values,
      llvm::StringRef abiLabel,
      llvm::function_ref<mlir::LogicalResult(mlir::Type, mlir::ValueRange)>
          emitMember);
  // Best-effort liveness pin of a probe operand past a raw-word call: owned
  // MATERIALIZATION-CREATED payloads are consumed by a release-after (which
  // is also the pinning use); payloads that alias `source` (the operand's
  // py-level bundle) and borrowed payloads get a neutral manifest-method use
  // when one conforms (borrowed entry arguments outlive the call anyway).
  mlir::LogicalResult pinProbeOperandLiveness(mlir::Operation *op,
                                              const RuntimeBundle &payload,
                                              const RuntimeBundle *source);
  mlir::LogicalResult pinContainerLiveness(mlir::Operation *op,
                                           const RuntimeBundle &container,
                                           bool insertAfterOp = false);
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
  // Bind an element that ALREADY carries a reference. Same as the above minus
  // the retain: the marker still goes on, so the release still comes.
  //
  // No container fallback, and that is not an omission: the fallback exists to
  // retry a retain `atOperation` for elements whose defining ops are spread
  // across blocks, and an element that arrives owned comes straight out of one
  // runtime call. If its anchor cannot be found the frame has no way to release
  // it, so this REFUSES rather than binding a value nothing will free.
  mlir::LogicalResult bindOwnedEvidenceValue(mlir::Operation *op,
                                             mlir::Value resultValue,
                                             llvm::StringRef label,
                                             const RuntimeValue &value);
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
                              bool raiseOnMiss = true,
                              const RuntimeBundle *missingKeyForRepr = nullptr);
  // Both take the iterator bundle by value: they bind results into
  // `valueBundles` (a DenseMap) mid-lowering, which invalidates references
  // into the map.
  // Rank-1 memref view over a payload (inline descriptor assembly;
  // borrow-only). The pointer form is the assembly; the word form widens an
  // integer first and is the one to count -- see BoxLayout.cpp.
  static mlir::Value memrefFromBoxPointer(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value pointer,
                                          mlir::Value sizeWord,
                                          mlir::MemRefType type);
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
  // str instance of the uniform dispatch (print's conversion over erased
  // boxes).
  mlir::LogicalResult generateBoxedStrHook();
  // hash instance of the uniform dispatch: class id -> the manifest
  // `__hash__` returning the i64 hash word (dict/set probing over erased
  // keys).
  mlir::LogicalResult generateBoxedHashHook();
  // Source-class `__hash__` adapters for the boxed hash dispatch: a compiled
  // `def __hash__(self) -> int` returns the boxed-int ABI, which the uniform
  // i64 dispatch cannot call directly.
  mlir::LogicalResult synthesizeSourceClassHashAdapters();
  // Binary (same-class two-receiver) variant of the uniform dispatch:
  // `(ptr lhs, ptr rhs, i64 class_id) -> (results..., i1 handled)`; callees
  // take their self shape twice.
  mlir::LogicalResult generateBoxedBinaryMethodHook(
      llvm::StringRef hookName,
      llvm::function_ref<bool(mlir::func::FuncOp)> selects,
      mlir::TypeRange calleeResultTypes,
      llvm::StringRef sourceClassMethodName = "");
  // eq instance of the binary dispatch (dict/set key equality over erased
  // keys).
  mlir::LogicalResult generateBoxedEqHook();
  // lt instance of the binary dispatch (sort/ordering over erased values).
  mlir::LogicalResult generateBoxedLtHook();
  mlir::LogicalResult lowerListEvidenceNext(py::NextOp op,
                                            RuntimeBundle iterator);
  // True when `op` sits in a different block than the one defining the
  // container's physical storage (SpecialMethodOps.cpp).
  static bool crossesStorageDefiningBlock(mlir::Operation *op,
                                          const RuntimeBundle &bundle);
  // Drops a mutable container's compile-time contents evidence at an op the
  // walk cannot answer from it, and mirrors the drop into a field-alias
  // owner's cache (SpecialMethodOps.cpp).
  bool demoteCrossBlockContainerEvidence(mlir::Operation *op,
                                         mlir::Value containerValue);
  // Applies the rule above to every operand of `op` (SpecialMethodOps.cpp).
  void demoteCrossBlockContainerOperandEvidence(mlir::Operation *op);
  // Records that a value this op stored into a slot is now shared with a
  // holder (SpecialMethodOps.cpp).
  void markAbsorbedContainerAsShared(mlir::Operation *op);
  // Loop-body generator state-machine transform (GeneratorStateMachine.cpp).
  //
  // Suspension lane ABI (rfc/stdlib-semantics.md R3): a lane is one logical
  // value crossing the suspension boundary. Control lanes (state, inject,
  // has, hasret, arguments, frame slots in the int tier) ride the raw
  // (i64, i1) evidence pair. Value-bearing lanes may instead carry an
  // object-family value: its concrete physical span (runtimeValueTypesFor),
  // plus the trailing (i64, i1) evidence pair when the contract is
  // builtins.int. Ownership crosses the boundary through materialized
  // contracts (ly.ownership.owned_results on suspend results,
  // ly.ownership.transfer_args on resume arguments), which is what the
  // affine-ownership verifier's generator-frame rule consumes.
  struct GeneratorResumeLane {
    // Runtime contract name; empty for a control lane (pure (i64, i1) pair).
    std::string contract;
    // builtins.int value lane: physical span + trailing evidence pair.
    bool isInt = false;
    // types.NoneType lane: dead immortal placeholders cross the boundary.
    bool isNone = false;
    unsigned physicalCount = 0;
    // The contract's physical parts, carried on the lane rather than looked up
    // from the manifest on each use.
    // ⛔ Why NOT re-ask `manifest.valueShape(contract)` at every use site (which
    // is what `generatorLanePhysicalTypes` did): a SOURCE class has no manifest
    // shape at all -- its layout is computed from its ClassOp -- so the lookup
    // silently answered "no parts" for exactly the contracts this lane needs to
    // carry, and every generator yielding a user class fell back to the
    // int-only inline tier (seven probes, tests/probe/rebind_gen_w*.py).
    llvm::SmallVector<mlir::Type, 4> physicalTypes;
    bool isControl() const { return contract.empty(); }
  };
  struct GeneratorResumeInfo {
    std::string cloneName;
    unsigned frameWidth = 0;
    unsigned argumentCount = 0;
    // One lane per generator argument. Int arguments keep the legacy raw
    // (i64, i1) evidence pair through the drivers; object-contract arguments
    // ride their physical span (the generator retains them at creation and
    // the drop finalizer releases them, so the frame keeps its sources alive
    // however long the object outlives the call site).
    llvm::SmallVector<GeneratorResumeLane, 4> argumentLanes;
    // The yielded-value lane (result index 2 of the resume clone). Control
    // lane in the legacy int tier; object-family for boxed yields.
    GeneratorResumeLane valueLane;
    // Values live across a yield, one lane each. Lanes are grouped per
    // contract (lexicographic order) sized by the maximum same-contract live
    // count over all yields, so every suspension state maps its live values
    // onto type-stable lanes. The generator object absorbs each lane's
    // ownership between suspensions (frame words in its storage), which the
    // frame store/load helpers transfer in and out.
    llvm::SmallVector<GeneratorResumeLane, 8> frameLanes;
    // Lazily synthesized driver functions (see GeneratorStateMachine.cpp):
    // step resumes once and reports (has, value, ret); advance additionally
    // raises StopIteration on exhaustion; throw/close inject exceptions at
    // the suspension point through the EH TLS slot.
    std::string stepName;
    std::string advanceName;
    std::string throwName;
    std::string closeName;
    // Drop finalizer (RAII): runs close semantics (GeneratorExit into the
    // body so finally blocks execute) and releases surviving frame lanes
    // when the generator's refcount reaches zero without an explicit
    // close(). Dispatched per target id by the module drop hook.
    std::string finalizeName;
  };
  llvm::StringMap<GeneratorResumeInfo> generatorResumeClones;
  // Lane ABI helpers (GeneratorStateMachine.cpp). The resume-clone lookup
  // maps a clone function back to its GeneratorResumeInfo through the
  // kPrimitiveI64CloneAttr original-name attribute.
  GeneratorResumeInfo *generatorResumeInfoForClone(mlir::func::FuncOp clone);
  mlir::FailureOr<GeneratorResumeLane>
  computeGeneratorResumeLane(mlir::Operation *op, mlir::Type type);
  // A contract's suspension-lane parts, or nullopt when the contract cannot
  // ride a lane. Quiet by construction: the eligibility scan asks about every
  // yield in the module and a contract that cannot ride a lane is a fallback,
  // not a program error.
  std::optional<llvm::SmallVector<mlir::Type, 4>>
  generatorLaneParts(mlir::Operation *op, mlir::Type type) const;
  llvm::SmallVector<mlir::Type, 6>
  generatorLanePhysicalTypes(const GeneratorResumeLane &lane) const;
  // Immortal dead placeholders for a lane's physical span: release-safe
  // (immortal refcount) values that cross non-yield suspend exits so the
  // owned-result contract stays dischargeable on every path.
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
  materializeGeneratorDeadLaneValues(mlir::Operation *op,
                                     const GeneratorResumeLane &lane);
  // Append the physical return operands for one suspend lane (Returns.cpp
  // generator branch). Emits the pair-only materialization guard for int
  // lanes and dead placeholders for None-typed operands on object lanes.
  // `forceRetain` adds a reference even for owned bundles — used when the
  // same SSA value crosses in two lanes (yielded AND live across the yield),
  // where each lane must carry its own token.
  mlir::LogicalResult appendGeneratorLaneReturnOperands(
      mlir::func::ReturnOp op, const GeneratorResumeLane &lane,
      const RuntimeBundle &bundle, llvm::SmallVectorImpl<mlir::Value> &operands,
      bool forceRetain = false);
  // Frame words a lane occupies in the generator storage: (raw, valid) for
  // int lanes, then (pointer, size) per physical part.
  unsigned generatorLaneFrameWords(const GeneratorResumeLane &lane) const;
  // Identity function with an owned-results contract: roots a frame lane's
  // incoming ownership at a call so the refcount inserter and the affine
  // verifier track it like any produced resource (continuation block
  // arguments alone are invisible to both).
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorClaimFunction(mlir::Operation *op,
                                    const GeneratorResumeLane &lane);
  // Store/load one frame lane's physical span to/from the generator
  // storage words. Store consumes the span (ly.ownership.transfer_args);
  // load produces it (ly.ownership.owned_results) and zeroes the slot, so
  // ownership lives in exactly one place at any time.
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorFrameStoreFunction(mlir::Operation *op,
                                         const GeneratorResumeLane &lane);
  // Retaining counterpart for the creation-site argument persist. Same body,
  // but NO ly.ownership.transfer_args: the creation site has already emitted
  // an aggregate retain for the slot, and its own handle stays live (the
  // Python local is readable after the generator is built).
  // Why not reuse the transferring store here: one helper cannot carry both
  // effects, and declaring transfer while the site also retains produces two
  // runtime references against one release obligation -- the caller's token
  // is absorbed with no release placed, so the retain's reference is owned by
  // nobody and leaks once per generator built.
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorArgumentStoreFunction(mlir::Operation *op,
                                            const GeneratorResumeLane &lane);
  // Shared body for the two span stores above; `transferring` selects which
  // aggregate effect the symbol declares.
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorSpanStoreFunction(mlir::Operation *op,
                                        const GeneratorResumeLane &lane,
                                        llvm::StringRef name,
                                        bool transferring);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorFrameLoadFunction(mlir::Operation *op,
                                        const GeneratorResumeLane &lane);
  // Entry seeding for a resume clone with mixed lanes: control lanes ride
  // (i64, i1) pairs, frame lanes their physical spans (+ pair for int).
  mlir::LogicalResult
  seedGeneratorResumeCloneEntry(mlir::func::FuncOp function,
                                mlir::ArrayRef<mlir::Type> logicalTypes,
                                GeneratorResumeInfo &info);
  // Storage word offset of each frame lane (after the header words and the
  // argument words stored at creation for the drop finalizer).
  llvm::SmallVector<unsigned, 8>
  generatorFrameLaneWordOffsets(const GeneratorResumeInfo &info) const;
  // Argument-lane ABI helpers. Drivers pass int arguments as the legacy
  // (i64, i1) evidence pair and object arguments as their physical span.
  llvm::SmallVector<mlir::Type, 6>
  generatorArgumentPhysicalTypes(const GeneratorResumeLane &lane) const;
  unsigned
  generatorArgumentPhysicalCount(const GeneratorResumeInfo &info) const;
  // Storage words one argument occupies ((raw, valid) for int, box words per
  // physical part otherwise) and each argument's storage word offset.
  unsigned generatorArgumentFrameWords(const GeneratorResumeLane &lane) const;
  llvm::SmallVector<unsigned, 8>
  generatorArgumentWordOffsets(const GeneratorResumeInfo &info) const;
  // Resume-site operand assembly from the creation-site source bundles.
  mlir::LogicalResult appendGeneratorArgumentOperands(
      mlir::Operation *op, const GeneratorResumeInfo &info,
      llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> sources,
      llvm::SmallVectorImpl<mlir::Value> &operands);
  // Driver-side forwarding of the entry block's argument physicals.
  void appendGeneratorArgumentEntryOperands(
      mlir::Block *entry, const GeneratorResumeInfo &info,
      llvm::SmallVectorImpl<mlir::Value> &operands) const;
  // Borrowing span load for the finalizer: reconstructs an object argument
  // from its storage words WITHOUT zeroing them (unlike the frame-lane load,
  // ownership stays with the generator until the finalizer's release pass).
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorArgumentLoadFunction(mlir::Operation *op,
                                           const GeneratorResumeLane &lane);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorFinalizeFunction(GeneratorResumeInfo &info);
  // Per-program drop hook: patches the generator deallocator so a
  // release-to-zero dispatches the storage's target id to the matching
  // finalizer before the storage is freed (rfc/stdlib-semantics.md R3:
  // drop/close releases the absorbed frame values; CPython: __del__ closes).
  mlir::LogicalResult generateGeneratorDropHook();
  mlir::LogicalResult buildGeneratorResumeCloneSignatures();
  mlir::LogicalResult buildGeneratorResumeBodies();
  // Inline statically-bound `yield from inner(...)` delegations into the
  // resume clone (PEP 380 by frame merging). Returns false when a delegation
  // shape is not inlinable and the body must fall back to the legacy inline
  // dispatch.
  mlir::FailureOr<bool> inlineDelegatedYieldFroms(mlir::func::FuncOp clone);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorStepFunction(mlir::Operation *op,
                                   GeneratorResumeInfo &info);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorAdvanceFunction(mlir::Operation *op,
                                      GeneratorResumeInfo &info);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorThrowFunction(mlir::Operation *op,
                                    GeneratorResumeInfo &info);
  mlir::FailureOr<mlir::func::FuncOp>
  getOrCreateGeneratorCloseFunction(mlir::Operation *op,
                                    GeneratorResumeInfo &info);
  mlir::FailureOr<SourceGeneratorResumeResult> emitStateMachineGeneratorResume(
      mlir::Operation *op, const RuntimeBundle &iterator,
      GeneratorResumeInfo &info, bool useCurrentInsertionPoint = false,
      std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence = std::nullopt,
      bool raiseWhenExhausted = false);
  mlir::LogicalResult
  lowerStateMachineGeneratorThrow(py::CallOp op, const RuntimeBundle &receiver,
                                  GeneratorResumeInfo &info,
                                  llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult
  lowerStateMachineGeneratorClose(py::CallOp op, const RuntimeBundle &receiver,
                                  GeneratorResumeInfo &info);
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
  mlir::LogicalResult lowerComplexConstant(py::ComplexConstantOp op);
  mlir::LogicalResult lowerBoolConstant(py::BoolConstantOp op);
  mlir::LogicalResult lowerNone(py::NoneOp op);
  mlir::LogicalResult lowerCastFromPrim(py::CastFromPrimOp op);
  mlir::LogicalResult lowerCastToPrim(py::CastToPrimOp op);
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
                                               const RuntimeBundle &exception);
  mlir::LogicalResult emitSetCurrentCause(mlir::Operation *op,
                                          const RuntimeBundle &cause);
  mlir::LogicalResult lowerRaise(py::RaiseOp op);
  mlir::LogicalResult lowerRaiseCurrent(py::RaiseCurrentOp op);
  mlir::LogicalResult lowerExceptMatch(py::ExceptMatchOp op);
  mlir::LogicalResult lowerExceptCurrentMatch(py::ExceptCurrentMatchOp op);
  mlir::LogicalResult lowerExceptCurrentValue(py::ExceptCurrentValueOp op);
  mlir::LogicalResult lowerStarBegin(py::StarBeginOp op);
  mlir::LogicalResult lowerExceptStarMatch(py::ExceptStarMatchOp op);
  mlir::LogicalResult lowerStarCollect(py::StarCollectOp op);
  mlir::LogicalResult lowerStarBodyEnd(py::StarBodyEndOp op);
  mlir::LogicalResult lowerStarFinish(py::StarFinishOp op);
  mlir::LogicalResult emitTracebackFrame(mlir::Operation *op,
                                         bool stashCurrentException = true);
  mlir::LogicalResult lowerCall(py::CallOp op);
  mlir::LogicalResult lowerBoundMethodCall(py::CallOp op,
                                           const RuntimeBundle &receiver,
                                           llvm::StringRef methodName);
  mlir::LogicalResult
  lowerRuntimeListPop(py::CallOp op, const RuntimeBundle &receiver,
                      llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::LogicalResult
  lowerRuntimeListInsert(py::CallOp op, const RuntimeBundle &receiver,
                         llvm::ArrayRef<const RuntimeBundle *> sources);
  mlir::Value boundMethodArgumentValue(py::CallOp op, unsigned position) const;
  mlir::FailureOr<mlir::Value> transientPayloadBox(mlir::Operation *op,
                                                   const RuntimeBundle &payload,
                                                   bool ownsPayload);
  void demoteSequenceEvidence(py::CallOp op, const RuntimeBundle &receiver);
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
  mlir::LogicalResult consumeFunctionTargetCallResult(
      mlir::Operation *op, llvm::StringRef targetName, mlir::func::CallOp call,
      mlir::Type expectedResult, llvm::ArrayRef<const RuntimeBundle *> sources,
      bool applyReturnedSummaries, llvm::StringRef abiLabel,
      RuntimeBundle &result);
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
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
      const RuntimeBundle *callableObject = nullptr);
  mlir::LogicalResult materializeDefaultArgument(
      py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
      unsigned index, mlir::Type parameterType,
      llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
      const RuntimeBundle *&source,
      const RuntimeBundle *callableObject = nullptr);
  mlir::LogicalResult
  materializeArityObject(mlir::Operation *op, mlir::Type contract,
                         std::uint64_t arity, RuntimeBundle &bundle,
                         mlir::ArrayRef<RuntimeValue> elements = {},
                         llvm::ArrayRef<std::string> keys = {},
                         llvm::ArrayRef<const RuntimeBundle *> elementBundles =
                             {},
                         llvm::ArrayRef<mlir::Value> logicalSources = {});
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
  mlir::LogicalResult lowerIs(py::IsOp op);
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
  lowerSourceGeneratorDunderNext(py::CallOp op, const RuntimeBundle &receiver,
                                 llvm::ArrayRef<const RuntimeBundle *> sources);
  // Shared tail of send/__next__: resume once, raise StopIteration when the
  // body finished, and bind the yielded int as the call result.
  mlir::LogicalResult lowerSourceGeneratorAdvance(
      py::CallOp op, const RuntimeBundle &receiver,
      std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence);
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
  // "May this argument's edges be spliced now?" -- asked by both the deferral
  // and the drain, which is why it is one function.
  bool hasPrecedingSiblingInFlight(mlir::BlockArgument argument) const;
  mlir::LogicalResult
  lowerControlFlowBlockArgument(mlir::Operation *op,
                                mlir::BlockArgument argument);
  mlir::LogicalResult spliceControlFlowBlockArgumentEdges(
      mlir::Operation *op, mlir::BlockArgument argument,
      llvm::ArrayRef<mlir::Type> physicalTypes, bool primitiveIntLane);
  mlir::LogicalResult drainDeferredControlFlowExpansions();
  mlir::LogicalResult lowerRuntimeValueSelect(mlir::arith::SelectOp select);
  llvm::SmallVector<mlir::BlockArgument, 16>
  logicalBlockArgumentsHighestIndexFirst() const;
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
  // Contracts whose runtime layout is being expanded right now. A layout that
  // re-enters itself has no finite expansion (CallableABI.cpp).
  mutable llvm::DenseSet<mlir::Type> expandingContracts;
  mutable std::optional<llvm::StringSet<>> subclassedContracts;
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
  // Per-clone `memref<1xi64>` slot holding 1 while the clone's raw lane still
  // tracks the true Python values, 0 once some step discarded a validity bit.
  llvm::DenseMap<mlir::Operation *, mlir::Value>
      primitiveI64CloneSpeculationFlags;
  llvm::StringMap<std::int64_t> functionTargetIds;
  llvm::DenseMap<mlir::Block *, std::int64_t> tryHandlerIds;
  llvm::SmallVector<CallableLogicalEntryArgs, 8> callableLogicalEntryArgCounts;
  // Insertion-ordered: the drop/erase passes below walk it, and both a
  // membership test and a stable order are needed. Two containers held this
  // before -- a vector and a parallel DenseSet -- and every producer had to
  // remember to feed both.
  llvm::SmallSetVector<mlir::BlockArgument, 16>
      controlFlowLogicalBlockArguments;
  llvm::DenseSet<mlir::Value> controlFlowBlockArgumentsInProgress;
  llvm::SmallVector<ControlFlowDeferredExpansion, 4>
      controlFlowDeferredExpansions;
  std::int64_t nextFunctionTargetId = 1;
  std::int64_t nextTryHandlerId = 1;
  llvm::SmallVector<mlir::Operation *, 32> erase;
};

// Peels class-upcast / refine / protocol-view wrappers off a value: summary
// and await lowering must see the underlying object identity.
inline mlir::Value stripReturnedObjectView(mlir::Value value) {
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def || def->getNumOperands() != 1 || def->getNumResults() != 1)
      return value;
    llvm::StringRef name = def->getName().getStringRef();
    if (name != "py.class.upcast" && name != "py.class.refine" &&
        name != "py.protocol.view")
      return value;
    value = def->getOperand(0);
  }
  return value;
}

// Charge to `container` every slot-absorption retain emitted since `anchor`,
// so an ownership walk can name the `parent` of `aggregate(parent, path)`.
// Defined in Core/CollectionPayload.cpp beside the rule it serves.
//
// ⭐ Shared rather than file-local because the dict literal has TWO lowerings
// and only one of them lived there: an all-static-string-key literal fills its
// slots directly, and one non-static key sends the whole thing down the
// `setitem_box` probe path in Ops/PackAndBindingOps.cpp. Retains from the
// second went unparented, which the affine walk then counted in
// `state.retained` -- part of its visited-state key -- so `{i: 1}` in nested
// loops never closed the fixpoint.
void chargeSlotRetainsToParent(mlir::OpBuilder &builder, mlir::Block *block,
                               mlir::Operation *anchor,
                               const RuntimeBundle &container);
// The op the builder would insert after, or null at a block's beginning.
mlir::Operation *insertionAnchor(mlir::OpBuilder &builder);

// The Callable contract a lowered function carries; null for declarations
// and non-callable functions. Dozens of walks used to re-spell this
// attribute lookup inline.
inline py::CallableType callableTypeOf(mlir::func::FuncOp function) {
  if (!function || function.isDeclaration())
    return {};
  auto callableAttr = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  return mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
}

// Iterates `step` until it stops reporting changes — the driver behind every
// returned-* summary computation (they refine each other through the summary
// maps, so a single pass under-approximates).
template <typename Step> inline void runToFixpoint(Step &&step) {
  bool changed = true;
  while (changed) {
    changed = false;
    step(changed);
  }
}

inline bool isCoroutineLikeResultType(mlir::Type type) {
  if (runtimeContractName(type) == "types.CoroutineType")
    return true;
  auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type);
  return protocol && protocol.getProtocolName() == "Coroutine";
}

inline bool isAwaitIteratorLikeResultType(mlir::Type type) {
  std::string contract = runtimeContractName(type);
  if (contract == "types.CoroutineAwaitIterator" ||
      contract == "_asyncio.FutureIter" || contract == "_asyncio.TaskIter")
    return true;
  auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type);
  return protocol && protocol.getProtocolName() == "Generator";
}

inline mlir::func::FuncOp
getOrCreatePrivateFunction(mlir::ModuleOp module, mlir::OpBuilder &builder,
                           llvm::StringRef name, mlir::FunctionType type) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      mlir::func::FuncOp::create(builder, module.getLoc(), name, type);
  function.setPrivate();
  return function;
}

inline mlir::func::FuncOp
getOrCreateDiscardCurrentException(mlir::ModuleOp module,
                                   mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(module, builder,
                                    "LyEH_DiscardCurrentException",
                                    builder.getFunctionType({}, {}));
}

inline mlir::func::FuncOp
getOrCreateRethrowCurrent(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(module, builder, "LyEH_RethrowCurrent",
                                    builder.getFunctionType({}, {}));
}

// A raise that happens while another exception is being handled moves the
// handled exception into the raised exception's __context__ chain instead of
// releasing it (CPython implicit exception chaining).
inline mlir::func::FuncOp
getOrCreateStashCurrentAsContext(mlir::ModuleOp module,
                                 mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(module, builder,
                                    "LyEH_StashCurrentAsContext",
                                    builder.getFunctionType({}, {}));
}

inline mlir::func::FuncOp
getOrCreateSetCurrentSuppress(mlir::ModuleOp module, mlir::OpBuilder &builder) {
  return getOrCreatePrivateFunction(module, builder, "LyEH_SetCurrentSuppress",
                                    builder.getFunctionType({}, {}));
}

inline mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                              bool value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

inline bool sameRuntimeValueIdentity(const RuntimeValue &lhs,
                                     const RuntimeValue &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return false;
  if (lhs.values.empty())
    return false;
  // Ownership rewrapping (retain markers) must not break identity: compare
  // the values underneath any identity-cast markers.
  for (auto [left, right] : llvm::zip(lhs.values, rhs.values))
    if (ownership::underlyingObjectValue(left) !=
        ownership::underlyingObjectValue(right))
      return false;
  return true;
}

} // namespace py::lowering
