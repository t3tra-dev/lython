#pragma once

#include "Runtime/Model/Contracts.h"
#include "Ownership.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace py::lowering {

inline constexpr llvm::StringLiteral kFutureResultSlot{"asyncio.future.result"};
inline constexpr llvm::StringLiteral kFutureExceptionSlot{
    "asyncio.future.exception"};
inline constexpr llvm::StringLiteral kFutureCancelMessageSlot{
    "asyncio.future.cancel_message"};
inline constexpr llvm::StringLiteral kFutureCancelledFlag{
    "asyncio.future.cancelled"};
inline constexpr llvm::StringLiteral kCoroutineAwaitConsumedFlag{
    "asyncio.coroutine.await_consumed"};
inline constexpr llvm::StringLiteral kAsyncioSleepDelaySlot{
    "asyncio.sleep.delay"};
inline constexpr llvm::StringLiteral kAsyncioSleepResultSlot{
    "asyncio.sleep.result"};
inline constexpr llvm::StringLiteral kAsyncioSleepZeroDelayFlag{
    "asyncio.sleep.zero_delay"};
inline constexpr llvm::StringLiteral kAsyncioSleepTimerPendingFlag{
    "asyncio.sleep.timer_pending"};
inline constexpr llvm::StringLiteral kAsyncioSleepTimerScheduledFlag{
    "asyncio.sleep.timer_scheduled"};
inline constexpr llvm::StringLiteral kAsyncioSleepLoopSlot{
    "asyncio.sleep.loop"};

struct RuntimeValue {
  mlir::Type contract;
  llvm::SmallVector<mlir::Value, 4> values;
  ownership::OwnershipKind ownership =
      ownership::OwnershipKind::NonObject;

  static RuntimeValue object(mlir::Type contract, mlir::ValueRange values,
                             bool ownsObject = true);
  static RuntimeValue
  objectWithOwnership(mlir::Type contract, mlir::ValueRange values,
                      ownership::OwnershipKind ownership);
  std::string contractName() const;
  RuntimeValue
  withOwnership(ownership::OwnershipKind ownership) const;
  RuntimeValue withLogicalOwnership(bool ownsObject) const;
};

struct RuntimeCallableAlternative {
  std::string functionTarget;
  llvm::SmallVector<RuntimeValue, 4> closureValues;
};

struct RuntimeObjectEvidence {
  llvm::StringMap<RuntimeValue> slots;
  llvm::StringSet<> flags;

  const RuntimeValue *slot(llvm::StringRef name) const;
  void setSlot(llvm::StringRef name, const RuntimeValue &value);
  void eraseSlot(llvm::StringRef name);
  bool hasFlag(llvm::StringRef name) const;
  void setFlag(llvm::StringRef name);
  void eraseFlag(llvm::StringRef name);
};

struct RuntimePrimitiveI64Evidence {
  mlir::Value value;
  mlir::Value valid;
};

// Buffer evidence is the lowered counterpart of Python's buffer protocol. It
// deliberately carries only facts that can be proven statically by the
// producer: a native base address, byte length, mutability, contiguity, and
// explicit keepalive owners. ctypes.from_buffer consumes the writable view;
// ctypes.from_buffer_copy consumes any readable view and materializes owned
// ctypes storage.
struct RuntimeBufferEvidence {
  mlir::Value addressValue;
  mlir::Value addressValid;
  mlir::Value byteLength;
  mlir::Value byteLengthValid;
  llvm::SmallVector<RuntimeValue, 2> keepAlive;
  bool readable = false;
  bool writable = false;
  bool cContiguous = false;
};

// Static ctypes evidence mirrors CPython's _CData boundary without copying its
// dynamic GC bookkeeping. `storageAddressValue` is the b_ptr-like address of
// the C memory block when the _CData storage exists. `addressValue` is the
// native pointer value used when this evidence flows as a pointer argument
// (identical to storageAddressValue for scalar cells, pointee address for
// pointer objects). `keepAlive` represents explicit _objects/_b_base_ edges,
// and the provenance/lifetime pair tells native-call lowering whether the
// pointer is a call-region borrow, an owned native cell, or an external view.
struct RuntimeCtypesEvidence {
  enum class Kind { Module, Cell, Pointer, Library, Symbol, FieldDescriptor };
  enum class Provenance {
    None,
    NativeCell,
    CallRegionBorrow,
    ExternalAddress,
    Cast,
    BufferView,
    CallbackThunk
  };
  enum class Lifetime { Unknown, CallRegion, Owner, External, Static };

  Kind kind = Kind::Cell;
  Provenance provenance = Provenance::None;
  Lifetime lifetime = Lifetime::Unknown;
  std::string ctypeName;
  mlir::Type ctype;
  std::string libraryName;
  std::string abi;
  std::string symbolName;
  std::string fieldName;
  std::string fieldType;
  std::uint64_t fieldOffset = 0;
  std::uint64_t fieldSize = 0;
  llvm::SmallVector<std::string, 8> argTypes;
  std::optional<std::string> resultType;
  std::string pointeeType;
  mlir::Value pointeeScalarValue;
  mlir::Value pointeeScalarValid;
  mlir::Value scalarValue;
  mlir::Value scalarValid;
  mlir::Value storageAddressValue;
  mlir::Value storageAddressValid;
  mlir::Value addressValue;
  mlir::Value addressValid;
  llvm::SmallVector<RuntimeValue, 2> keepAlive;
  bool processLibrary = false;
  bool callRegionBorrow = false;
  bool ownsNativeStorage = false;
  bool materializedObject = false;
};

struct RuntimeBundle {
  enum class Kind { Object, Aggregate, BuiltinCallable, TypeObject };

  Kind kind = Kind::Object;
  mlir::Type contract;
  mlir::Type instanceContract;
  RuntimeValue objectValue;
  mlir::Value fieldAliasOwner;
  std::string fieldAliasName;
  llvm::SmallVector<mlir::Value, 4> aggregateOperands;
  llvm::SmallVector<char, 4> aggregateUnpackedOperands;
  std::string binding;
  std::optional<std::string> literalText;
  std::string functionTarget;
  llvm::SmallVector<RuntimeValue, 4> closureValues;
  llvm::SmallVector<RuntimeCallableAlternative, 4> callableAlternatives;
  std::shared_ptr<RuntimeBundle> boundMethodReceiver;
  std::string boundMethodName;
  std::string coroutineTarget;
  llvm::SmallVector<RuntimeValue, 8> coroutineSources;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> coroutineSourceBundles;
  std::optional<RuntimePrimitiveI64Evidence> primitiveI64;
  std::optional<RuntimeBufferEvidence> buffer;
  std::optional<RuntimeCtypesEvidence> ctypes;
  RuntimeObjectEvidence objectEvidence;
  llvm::StringMap<std::shared_ptr<RuntimeBundle>> fieldBundles;
  std::shared_ptr<RuntimeBundle> boxedObject;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> sequenceElementBundles;
  llvm::SmallVector<RuntimeValue, 8> sequenceElements;
  llvm::SmallVector<std::int64_t, 8> sequenceIndices;
  std::uint64_t sequenceCapacity = 0;
  llvm::SmallVector<std::string, 8> mappingKeys;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> mappingKeyBundles;
  llvm::SmallVector<RuntimeValue, 8> mappingValues;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> mappingValueBundles;
  llvm::SmallVector<mlir::Value, 8> mappingPresent;
  std::uint64_t mappingCapacity = 0;

  static RuntimeBundle object(mlir::Type contract, mlir::ValueRange values);
  static RuntimeBundle
  objectWithOwnership(mlir::Type contract, mlir::ValueRange values,
                      ownership::OwnershipKind ownership);
  static RuntimeBundle aggregate(mlir::Type contract,
                                 mlir::ValueRange operands);
  static RuntimeBundle builtinCallable(mlir::Type contract,
                                       llvm::StringRef binding);
  static RuntimeBundle typeObject(mlir::Type typeContract,
                                  mlir::Type instanceContract);
  void copyEvidenceFrom(const RuntimeBundle &source);
  llvm::ArrayRef<mlir::Value> physicalValues() const;
  std::string contractName() const;
  std::string instanceContractName() const;
  void setObjectOwnership(ownership::OwnershipKind ownership);
  void setObjectLogicalOwnership(bool ownsObject);
  RuntimeBundle
  withObjectOwnership(ownership::OwnershipKind ownership) const;
};

inline bool hasFutureTerminalEvidence(const RuntimeBundle &future) {
  return future.objectEvidence.slot(kFutureResultSlot) ||
         future.objectEvidence.slot(kFutureExceptionSlot) ||
         future.objectEvidence.hasFlag(kFutureCancelledFlag);
}

inline bool hasAsyncioSleepEvidence(const RuntimeBundle &coroutine) {
  return coroutine.objectEvidence.slot(kAsyncioSleepResultSlot);
}

struct CallableLogicalEntryArgs {
  mlir::func::FuncOp function;
  unsigned count = 0;
};

// Primitive callable clones are derived from the Callable contract and use the
// lyrt.prim-shaped value/evidence pair directly. They are not separate Python
// object types.
inline constexpr llvm::StringLiteral kPrimitiveI64CloneAttr{
    "ly.primitive_i64_clone"};
inline constexpr llvm::StringLiteral kProtocolTemplateAttr{
    "ly.protocol_template"};
inline constexpr llvm::StringLiteral kProtocolSpecializationAttr{
    "ly.protocol_specialization_of"};

struct ControlFlowLogicalBlockArgumentABI {
  mlir::BlockArgument argument;
};

struct ReturnedCallableAlternativeSummary {
  std::string target;
  llvm::SmallVector<unsigned, 4> captureArgumentIndices;
};

struct ReturnedCallableSummary {
  llvm::SmallVector<ReturnedCallableAlternativeSummary, 4> alternatives;
};

struct ReturnedCoroutineSummary {
  std::string target;
  llvm::SmallVector<mlir::Type, 4> sourceContracts;
};

struct ReturnedObjectEvidenceSlot {
  std::string name;
  mlir::Type sourceContract;
};

struct ReturnedObjectEvidenceSummary {
  mlir::Type objectContract;
  unsigned resultIndex = 0;
  llvm::SmallVector<std::string, 4> flags;
  llvm::SmallVector<ReturnedObjectEvidenceSlot, 4> slots;
};

struct ReturnedStaticObjectSummary {
  mlir::Type objectContract;
  unsigned resultIndex = 0;
};

struct ReturnedValueSummary {
  llvm::SmallVector<unsigned, 4> argumentIndices;
};

struct PlannedKeywordArgument {
  std::string name;
  unsigned actualIndex = 0;
  mlir::Type type;
};

struct CallableArgumentPlan {
  llvm::SmallVector<std::optional<unsigned>, 8> fixedActuals;
  llvm::SmallVector<unsigned, 4> defaultedFixed;
  llvm::SmallVector<unsigned, 4> varargActuals;
  llvm::SmallVector<unsigned, 4> kwargActuals;
};

struct RuntimeArgumentEvidence {
  std::string functionTarget;
  llvm::SmallVector<mlir::Type, 4> closureValueTypes;
  std::string coroutineTarget;
  llvm::SmallVector<mlir::Type, 4> coroutineSourceTypes;

  bool empty() const {
    return functionTarget.empty() && closureValueTypes.empty() &&
           coroutineTarget.empty() && coroutineSourceTypes.empty();
  }
};

struct RuntimeArgumentEvidenceSet {
  llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;

  bool empty() const {
    return llvm::all_of(alternatives,
                        [](const RuntimeArgumentEvidence &evidence) {
                          return evidence.empty();
                        });
  }
};

struct CallableArgumentEvidenceABI {
  llvm::SmallVector<RuntimeArgumentEvidenceSet, 8> logicalArguments;

  bool empty() const {
    return llvm::all_of(logicalArguments,
                        [](const RuntimeArgumentEvidenceSet &evidence) {
                          return evidence.empty();
                        });
  }
};

struct StaticCallableInvocation {
  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<mlir::Type, 8> actualTypes;
  llvm::SmallVector<mlir::Value, 8> actualValues;
  llvm::SmallVector<PlannedKeywordArgument, 8> keywords;
};

struct CallableAggregateEvidenceCall {
  llvm::SmallVector<mlir::Type, 8> varargElementTypes;
  llvm::SmallVector<std::string, 8> kwargKeys;
  llvm::SmallVector<mlir::Type, 8> kwargValueTypes;
};

struct CallableAggregateEvidenceABI {
  std::optional<unsigned> varargLogicalIndex;
  llvm::SmallVector<std::int64_t, 8> varargElementIndices;
  llvm::SmallVector<mlir::Type, 8> varargElementTypes;
  std::optional<unsigned> kwargLogicalIndex;
  bool kwargIsFull = false;
  llvm::SmallVector<std::string, 8> kwargKeys;
  llvm::SmallVector<mlir::Type, 8> kwargValueTypes;
};

} // namespace py::lowering
