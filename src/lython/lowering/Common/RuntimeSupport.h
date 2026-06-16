#pragma once

#include "PyDialectTypes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace py {

struct RuntimeSymbols {
  static constexpr llvm::StringLiteral kHostPrintMemRefLine{"LyHost_PrintLine"};
  static constexpr llvm::StringLiteral kExceptionNew{"LyException_New"};
  static constexpr llvm::StringLiteral kExceptionDecRef{"LyException_DecRef"};
  static constexpr llvm::StringLiteral kEHThrowException{"LyEH_ThrowException"};
  static constexpr llvm::StringLiteral kEHRethrowCurrent{"LyEH_RethrowCurrent"};
  static constexpr llvm::StringLiteral kEHTakeCurrentDescriptor{
      "LyEH_TakeCurrentDescriptor"};
  static constexpr llvm::StringLiteral kTracebackPush{"LyTraceback_Push"};
  static constexpr llvm::StringLiteral kTracebackPop{"LyTraceback_Pop"};
  static constexpr llvm::StringLiteral kTracebackClear{"LyTraceback_Clear"};
  static constexpr llvm::StringLiteral kTracebackPrintMessage{
      "LyTraceback_PrintMessage"};
  static constexpr llvm::StringLiteral kLongFromI64{"LyLong_FromI64"};
  static constexpr llvm::StringLiteral kLongAsI64{"LyLong_AsI64"};
  static constexpr llvm::StringLiteral kLongAdd{"LyLong_Add"};
  static constexpr llvm::StringLiteral kLongSub{"LyLong_Sub"};
  static constexpr llvm::StringLiteral kLongMul{"LyLong_Mul"};
  static constexpr llvm::StringLiteral kLongCompare{"LyLong_Compare"};
  static constexpr llvm::StringLiteral kLongRepr{"LyLong_Repr"};
  static constexpr llvm::StringLiteral kLongDecRef{"LyLong_DecRef"};
  static constexpr llvm::StringLiteral kUnicodeFromBytes{"LyUnicode_FromBytes"};
  static constexpr llvm::StringLiteral kUnicodeAlloc{"__ly_unicode_alloc"};
  static constexpr llvm::StringLiteral kUnicodeConcat{"LyUnicode_Concat"};
  static constexpr llvm::StringLiteral kUnicodeConcat3{"LyUnicode_Concat3"};
  static constexpr llvm::StringLiteral kUnicodeCopy{"LyUnicode_Copy"};
  static constexpr llvm::StringLiteral kUnicodeEqBool{"LyUnicode_EqBool"};
  static constexpr llvm::StringLiteral kUnicodeLength{"LyUnicode_Length"};
  static constexpr llvm::StringLiteral kUnicodeCodepointLength{
      "LyUnicode_CodepointLength"};
  static constexpr llvm::StringLiteral kUnicodePrint{"LyUnicode_Print"};
  static constexpr llvm::StringLiteral kUnicodePrintLine{"LyUnicode_PrintLine"};
  static constexpr llvm::StringLiteral kUnicodeDecRef{"LyUnicode_DecRef"};
  static constexpr llvm::StringLiteral kUnicodeFromI64{"LyUnicode_FromI64"};
  static constexpr llvm::StringLiteral kIncRef{"Ly_IncRef"};
};

namespace OwnershipContractAttrs {
static constexpr llvm::StringLiteral kRetainArgs{"ly.ownership.retain_args"};
static constexpr llvm::StringLiteral kReleaseArgs{"ly.ownership.release_args"};
static constexpr llvm::StringLiteral kTransferArgs{
    "ly.ownership.transfer_args"};
static constexpr llvm::StringLiteral kOwnedResults{
    "ly.ownership.owned_results"};
static constexpr llvm::StringLiteral kBorrowedResults{
    "ly.ownership.borrowed_results"};
static constexpr llvm::StringLiteral kSetFieldValueArg{
    "ly.ownership.setfield_value_arg"};
static constexpr llvm::StringLiteral kSetFieldRetainArg{
    "ly.ownership.setfield_retain_arg"};
static constexpr llvm::StringLiteral kGetFieldBorrowArg{
    "ly.ownership.getfield_borrow_arg"};
static constexpr llvm::StringLiteral kGetFieldOwnedResult{
    "ly.ownership.getfield_owned_result"};
static constexpr llvm::StringLiteral kAggregateRetain{
    "ly.ownership.aggregate_retain"};
static constexpr llvm::StringLiteral kAggregateRelease{
    "ly.ownership.aggregate_release"};
static constexpr llvm::StringLiteral kLocalDestroy{
    "ly.ownership.local_destroy"};
static constexpr llvm::StringLiteral kFrameTransfer{
    "ly.ownership.frame_transfer"};
static constexpr llvm::StringLiteral kMemRefSlotTransfer{
    "ly.ownership.memref_slot_transfer"};
static constexpr llvm::StringLiteral kAggregateSlotLoad{
    "ly.ownership.aggregate_slot_load"};
static constexpr llvm::StringLiteral kAggregateSlotGroup{
    "ly.ownership.aggregate_slot_group"};
static constexpr llvm::StringLiteral kAggregateSlotComponent{
    "ly.ownership.aggregate_slot_component"};
static constexpr llvm::StringLiteral kAggregateSlotIndex{
    "ly.ownership.aggregate_slot_index"};
static constexpr llvm::StringLiteral kAggregateSlotPartIndex{
    "ly.ownership.aggregate_slot_part_index"};
static constexpr llvm::StringLiteral kNonObjectPointer{
    "ly.ownership.non_object_pointer"};
static constexpr llvm::StringLiteral kOwnedLocalObject{
    "ly.ownership.owned_local_object"};
static constexpr llvm::StringLiteral kObjectHeader{
    "ly.ownership.object_header"};
static constexpr llvm::StringLiteral kImmortalObject{
    "ly.ownership.immortal_object"};
static constexpr llvm::StringLiteral kObjectReleaseToZero{
    "ly.ownership.object_release_to_zero"};
static constexpr llvm::StringLiteral kObjectDeallocPart{
    "ly.ownership.object_dealloc_part"};
} // namespace OwnershipContractAttrs

namespace ControlFlowContractAttrs {
static constexpr llvm::StringLiteral kNoReturn{"ly.control.noreturn"};
} // namespace ControlFlowContractAttrs

namespace ThreadSafetyAttrs {
static constexpr llvm::StringLiteral kAtomicRole{"ly.atomic.role"};
static constexpr llvm::StringLiteral kAtomicOrdering{"ly.atomic.ordering"};
static constexpr llvm::StringLiteral kAtomicProvenance{"ly.atomic.provenance"};
static constexpr llvm::StringLiteral kAtomicMemRefComponent{
    "ly.atomic.memref_component"};
static constexpr llvm::StringLiteral kAtomicMemRefSlot{"ly.atomic.memref_slot"};
static constexpr llvm::StringLiteral kAtomicMemRefGroup{
    "ly.atomic.memref_group"};
static constexpr llvm::StringLiteral kAtomicContainerKind{
    "ly.atomic.container_kind"};
static constexpr llvm::StringLiteral kRetainPremise{"ly.atomic.retain_premise"};
static constexpr llvm::StringLiteral kOwnedTokenVerified{
    "ly.atomic.owned_token_verified"};
static constexpr llvm::StringLiteral kOwnedTokenProof{
    "ly.atomic.owned_token_proof"};

static constexpr llvm::StringLiteral kOrderingAcquire{"acquire"};
static constexpr llvm::StringLiteral kOrderingRelease{"release"};
static constexpr llvm::StringLiteral kOrderingAcqRel{"acq_rel"};
static constexpr llvm::StringLiteral kOrderingMonotonic{"monotonic"};
static constexpr llvm::StringLiteral kOrderingSeqCst{"seq_cst"};

static constexpr llvm::StringLiteral kPremiseOwnedToken{"owned-token"};
static constexpr llvm::StringLiteral kPremiseEntryBorrowed{"entry-borrowed"};
static constexpr llvm::StringLiteral kPremiseCapturedBorrowed{
    "captured-borrowed"};
static constexpr llvm::StringLiteral kPremiseLockedBorrow{"locked-borrow"};
static constexpr llvm::StringLiteral kPremiseAggregateBorrow{
    "aggregate-borrow"};
static constexpr llvm::StringLiteral kProofOwnershipVerifier{
    "ownership-verifier"};
static constexpr llvm::StringLiteral kProofClassFieldHelper{
    "class-field-helper"};
static constexpr llvm::StringLiteral kProvenanceMemRefDescriptor{
    "memref-descriptor"};

static constexpr llvm::StringLiteral kRoleContainerRefcountLoad{
    "container.refcount.load"};
static constexpr llvm::StringLiteral kRoleContainerRefcountRetain{
    "container.refcount.retain"};
static constexpr llvm::StringLiteral kRoleContainerRefcountRelease{
    "container.refcount.release"};
static constexpr llvm::StringLiteral kRoleObjectRefcountLoad{
    "object.refcount.load"};
static constexpr llvm::StringLiteral kRoleObjectRefcountInit{
    "object.refcount.init"};
static constexpr llvm::StringLiteral kRoleObjectRefcountRetain{
    "object.refcount.retain"};
static constexpr llvm::StringLiteral kRoleObjectRefcountRelease{
    "object.refcount.release"};
static constexpr llvm::StringLiteral kRoleObjectKindInit{"object.kind.init"};
static constexpr llvm::StringLiteral kRoleObjectPayloadInit{
    "object.payload.init"};
static constexpr llvm::StringLiteral kRoleObjectPayloadLoad{
    "object.payload.load"};
static constexpr llvm::StringLiteral kRoleContainerLockAcquire{
    "container.lock.acquire"};
static constexpr llvm::StringLiteral kRoleContainerLockRelease{
    "container.lock.release"};
static constexpr llvm::StringLiteral kRoleClassRefcountRetain{
    "class.refcount.retain"};
static constexpr llvm::StringLiteral kRoleClassRefcountRelease{
    "class.refcount.release"};
static constexpr llvm::StringLiteral kRoleClassRefcountLoad{
    "class.refcount.load"};
static constexpr llvm::StringLiteral kRoleClassLockAcquire{
    "class.lock.acquire"};
static constexpr llvm::StringLiteral kRoleClassLockRelease{
    "class.lock.release"};
static constexpr llvm::StringLiteral kRoleAsyncExceptionLoad{
    "async.exception.load"};
static constexpr llvm::StringLiteral kRoleAsyncExceptionStore{
    "async.exception.store"};
} // namespace ThreadSafetyAttrs

namespace ContainerSafetyAttrs {
static constexpr llvm::StringLiteral kRefcountInit{
    "ly.container.refcount_init"};
static constexpr llvm::StringLiteral kRefcountState{
    "ly.container.refcount_state"};
static constexpr llvm::StringLiteral kStateManaged{"managed"};
static constexpr llvm::StringLiteral kDescriptorGroup{
    "ly.container.descriptor_group"};
static constexpr llvm::StringLiteral kDescriptorKind{
    "ly.container.descriptor_kind"};
static constexpr llvm::StringLiteral kDescriptorComponent{
    "ly.container.descriptor_component"};
static constexpr llvm::StringLiteral kDescriptorData{
    "ly.container.descriptor_data"};
static constexpr llvm::StringLiteral kAccessGroup{"ly.container.access_group"};
static constexpr llvm::StringLiteral kAccessComponent{
    "ly.container.access_component"};
static constexpr llvm::StringLiteral kDeallocGroup{
    "ly.container.dealloc_group"};
static constexpr llvm::StringLiteral kDeallocComponent{
    "ly.container.dealloc_component"};
static constexpr llvm::StringLiteral kKindList{"list"};
static constexpr llvm::StringLiteral kKindTuple{"tuple"};
static constexpr llvm::StringLiteral kKindDict{"dict"};
static constexpr llvm::StringLiteral kComponentHeader{"header"};
static constexpr llvm::StringLiteral kComponentItems{"items"};
static constexpr llvm::StringLiteral kComponentKeys{"keys"};
static constexpr llvm::StringLiteral kComponentValues{"values"};
static constexpr llvm::StringLiteral kComponentStates{"states"};
static constexpr llvm::StringLiteral kComponentLock{"lock"};
} // namespace ContainerSafetyAttrs

namespace ClassSafetyAttrs {
static constexpr llvm::StringLiteral kHelperKind{"ly.class_helper.kind"};
static constexpr llvm::StringLiteral kHelperClass{"ly.class_helper.class"};
static constexpr llvm::StringLiteral kHelperFieldIndex{
    "ly.class_helper.field_index"};
static constexpr llvm::StringLiteral kHelperFieldCount{
    "ly.class_helper.field_count"};
static constexpr llvm::StringLiteral kHelperDirectRefcountFields{
    "ly.class_helper.direct_refcount_fields"};
static constexpr llvm::StringLiteral kHelperContainerFields{
    "ly.class_helper.container_fields"};
static constexpr llvm::StringLiteral kHelperDirectRefcountFieldIndices{
    "ly.class_helper.direct_refcount_field_indices"};
static constexpr llvm::StringLiteral kHelperContainerFieldIndices{
    "ly.class_helper.container_field_indices"};
static constexpr llvm::StringLiteral kBorrowedLocalField{
    "ly.class_helper.borrowed_local_field"};
static constexpr llvm::StringLiteral kPromoteFreshObject{
    "ly.class_helper.promote_fresh_object"};
static constexpr llvm::StringLiteral kPromoteLockInit{
    "ly.class_helper.promote_lock_init"};
static constexpr llvm::StringLiteral kPromoteRefcountInit{
    "ly.class_helper.promote_refcount_init"};
static constexpr llvm::StringLiteral kDeallocPart{"ly.class.dealloc_part"};
static constexpr llvm::StringLiteral kPayloadPart{"ly.class.payload_part"};
static constexpr llvm::StringLiteral kCarrierPack{"ly.class_carrier.pack"};
static constexpr llvm::StringLiteral kCarrierPart{"ly.class_carrier.part"};
static constexpr llvm::StringLiteral kCarrierLoad{"ly.class_carrier.load"};
static constexpr llvm::StringLiteral kCarrierPartRank{
    "ly.class_carrier.part_rank"};
static constexpr llvm::StringLiteral kCarrierPartElementWidth{
    "ly.class_carrier.part_element_width"};
static constexpr llvm::StringLiteral kCarrierPartStaticSize{
    "ly.class_carrier.part_static_size"};
static constexpr llvm::StringLiteral kKindIncref{"incref"};
static constexpr llvm::StringLiteral kKindDecref{"decref"};
static constexpr llvm::StringLiteral kKindDestroyLocal{"destroy_local"};
static constexpr llvm::StringLiteral kKindPromote{"promote"};
static constexpr llvm::StringLiteral kKindGetField{"getfield"};
static constexpr llvm::StringLiteral kKindSetField{"setfield"};
} // namespace ClassSafetyAttrs

namespace AsyncSafetyAttrs {
static constexpr llvm::StringLiteral kExceptionCell{"ly.async.exception_cell"};
static constexpr llvm::StringLiteral kExceptionCellReservation{
    "ly.async.exception_cell_reservation"};
static constexpr llvm::StringLiteral kExceptionCellConditionalStore{
    "ly.async.exception_cell_conditional_store"};
static constexpr llvm::StringLiteral kExceptionCellAllocated{
    "ly.async.exception_cell_allocated"};
static constexpr llvm::StringLiteral kExceptionCellFree{
    "ly.async.exception_cell_free"};
static constexpr llvm::StringLiteral kExceptionCellTransferArgs{
    "ly.async.exception_cell_transfer_args"};
static constexpr llvm::StringLiteral kExceptionCellPayloadStore{
    "ly.async.exception_cell_payload_store"};
static constexpr llvm::StringLiteral kRuntimeHandle{"ly.async.runtime_handle"};
static constexpr llvm::StringLiteral kRuntimeRefcountDelta{
    "ly.async.runtime_refcount_delta"};
static constexpr llvm::StringLiteral kRuntimeHandleBorrowArgs{
    "ly.async.runtime_handle_borrow_args"};
static constexpr llvm::StringLiteral kRuntimeHandleTransferArgs{
    "ly.async.runtime_handle_transfer_args"};
static constexpr llvm::StringLiteral kRuntimeHandleOwnedResults{
    "ly.async.runtime_handle_owned_results"};
static constexpr llvm::StringLiteral kRuntimeExecuteEntry{
    "ly.async.runtime_execute_entry"};
static constexpr llvm::StringLiteral kRuntimeAwaitExecute{
    "ly.async.runtime_await_execute"};
static constexpr llvm::StringLiteral kRuntimeErrorQuery{
    "ly.async.runtime_error_query"};
static constexpr llvm::StringLiteral kRuntimeValueStorage{
    "ly.async.runtime_value_storage"};
} // namespace AsyncSafetyAttrs

enum class AsyncArgProvenanceKind {
  RuntimeHandle,
  ExceptionCell,
};

struct AsyncArgProvenanceContract {
  std::string symbolName;
  unsigned argIndex;
  AsyncArgProvenanceKind kind;
};

struct NonObjectArgContract {
  std::string symbolName;
  unsigned argIndex;
};

struct MemRefAtomicContract {
  int64_t id;
  mlir::Location location;
  mlir::arith::AtomicRMWKind kind;
  int64_t value;
  mlir::StringAttr role;
  mlir::StringAttr ordering;
  mlir::StringAttr retainPremise;
  mlir::StringAttr group;
  mlir::StringAttr containerKind;
  mlir::StringAttr component;
  std::optional<int64_t> slot;
  bool ownedTokenVerified;
  mlir::StringAttr ownedTokenProof;
};

struct MemRefAggregateLoadContract {
  int64_t id;
  mlir::Location location;
  mlir::StringAttr group;
  mlir::StringAttr component;
  std::optional<int64_t> slot;
};

struct MemRefContainerAccessContract {
  int64_t id;
  mlir::Location location;
  bool store;
  mlir::StringAttr group;
  mlir::StringAttr component;
};

struct MemRefDeallocContract {
  int64_t id;
  mlir::Location location;
  mlir::StringAttr group;
  mlir::StringAttr component;
  mlir::StringAttr classPart;
  mlir::StringAttr objectPart;
  bool exceptionCellFree;
};

struct LoweredSafetyContracts {
  llvm::SmallVector<AsyncArgProvenanceContract> asyncArgs;
  llvm::SmallVector<NonObjectArgContract> nonObjectArgs;
  llvm::SmallVector<MemRefAtomicContract> memRefAtomics;
  llvm::SmallVector<MemRefAggregateLoadContract> aggregateLoads;
  llvm::SmallVector<MemRefContainerAccessContract> containerAccesses;
  llvm::SmallVector<MemRefDeallocContract> deallocs;
};

class PyLLVMTypeConverter;

namespace runtime::mlir_async {
struct Callee {
  static bool valueStorage(llvm::StringRef name);
  static bool known(llvm::StringRef name);
  static bool refcount(llvm::StringRef name);
  static bool createHandle(llvm::StringRef name);
  static bool isError(llvm::StringRef name);
  static bool awaitAndExecute(llvm::StringRef name);
  static bool executeEntry(llvm::StringRef name);
  static std::optional<int64_t> refcountDelta(llvm::StringRef name);
  static llvm::SmallVector<unsigned, 2>
  borrowedHandleOperands(llvm::StringRef name);
};
} // namespace runtime::mlir_async

namespace ownership::effect {
void retain(mlir::Operation *op, llvm::ArrayRef<int64_t> indices);
void release(mlir::Operation *op, llvm::ArrayRef<int64_t> indices);
void transfer(mlir::Operation *op, llvm::ArrayRef<int64_t> indices);
void ownedResults(mlir::Operation *op, llvm::ArrayRef<int64_t> indices);
} // namespace ownership::effect

namespace publication::borrow::Attr {
std::string name(unsigned argIndex);
} // namespace publication::borrow::Attr

namespace threadsafe {
struct Atomic {
  static void set(mlir::Operation *op, llvm::StringRef role,
                  llvm::StringRef ordering, llvm::StringRef retainPremise = {});
};

namespace memref {
struct Atomic {
  static void set(mlir::Operation *op, llvm::StringRef component,
                  std::optional<int64_t> slot, llvm::StringRef group = {},
                  llvm::StringRef containerKind = {});
};
} // namespace memref

struct Retain {
  static void premise(mlir::Operation *op, llvm::StringRef retainPremise);
  static void verifyOwnedToken(
      mlir::Operation *op,
      llvm::StringRef proof = ThreadSafetyAttrs::kProofOwnershipVerifier);
};
} // namespace threadsafe

namespace ownership::aggregate {
struct Slot {
  static void markLoad(mlir::Value value);
  static void markLoad(mlir::Value value, llvm::StringRef group,
                       llvm::StringRef component,
                       std::optional<int64_t> slot = std::nullopt);
  static void copyLoad(mlir::Value from, mlir::Value to);
  static void markStore(mlir::Operation *op);
  static void markStore(mlir::Operation *op, llvm::StringRef group,
                        llvm::StringRef component,
                        std::optional<int64_t> slot = std::nullopt);
};
} // namespace ownership::aggregate

namespace ownership {
struct Pointer {
  static void markNonObject(mlir::Value value);
};
} // namespace ownership

namespace container::Header {
void markAtomicResource(mlir::Operation *op, mlir::Value header,
                        std::optional<int64_t> slot);
} // namespace container::Header

namespace container::descriptor::Group {
std::string make(mlir::Operation *owner, llvm::StringRef kind);
} // namespace container::descriptor::Group

namespace container::descriptor::Component {
void mark(mlir::Value value, llvm::StringRef group, llvm::StringRef component);
} // namespace container::descriptor::Component

namespace async_runtime {

mlir::MemRefType getExceptionCellType(mlir::MLIRContext *ctx);
bool isExceptionCellType(mlir::Type type);
bool isLoweredExceptionCellType(mlir::Type type);

struct RuntimeHandle {
  static void mark(mlir::Value value);
  static void markArgument(mlir::Operation *funcLike, unsigned argIndex);
};

struct ExceptionCell {
  static void mark(mlir::Value value);
  static void markArgument(mlir::Operation *funcLike, unsigned argIndex);
  static bool hasProvenance(mlir::Value value);
  static mlir::Value load(mlir::Location loc, mlir::Value cell,
                          mlir::RewriterBase &rewriter);
  static mlir::Value isNull(mlir::Location loc, mlir::Value exception,
                            mlir::OpBuilder &builder);
  static mlir::LogicalResult storeFirst(
      mlir::Location loc, mlir::Value cell, mlir::Value exception,
      mlir::ModuleOp module, mlir::RewriterBase &rewriter,
      const PyLLVMTypeConverter &typeConverter,
      llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken);
  static mlir::LogicalResult releaseLoaded(
      mlir::Location loc, mlir::ModuleOp module, mlir::RewriterBase &rewriter,
      const PyLLVMTypeConverter &typeConverter, mlir::Value exception);
  static mlir::LogicalResult
  retainPayload(mlir::Location loc, mlir::ModuleOp module,
                mlir::OpBuilder &builder,
                const PyLLVMTypeConverter &typeConverter, mlir::Value exception,
                llvm::StringRef retainPremise);
  static mlir::FailureOr<mlir::Operation *>
  releasePayload(mlir::Location loc, mlir::ModuleOp module,
                 mlir::OpBuilder &builder,
                 const PyLLVMTypeConverter &typeConverter,
                 mlir::Value exception, bool aggregateLoaded = false);
  static mlir::LogicalResult free(mlir::Location loc, mlir::ModuleOp module,
                                  mlir::OpBuilder &builder,
                                  const PyLLVMTypeConverter &typeConverter,
                                  mlir::Value exceptionCell);
  static mlir::LogicalResult destroy(mlir::Location loc, mlir::ModuleOp module,
                                     mlir::RewriterBase &rewriter,
                                     const PyLLVMTypeConverter &typeConverter,
                                     mlir::Value exceptionCell);
  static mlir::LogicalResult destroyKnownEmpty(
      mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
      const PyLLVMTypeConverter &typeConverter, mlir::Value exceptionCell);
};

} // namespace async_runtime

namespace container::access::Contract {
bool has(mlir::Operation *op);
void copy(mlir::Operation *from, mlir::Operation *to);
void mark(mlir::Operation *op, mlir::Value header, mlir::Value target);
void collect(mlir::ModuleOp module,
             llvm::SmallVectorImpl<MemRefContainerAccessContract> &contracts);
mlir::LogicalResult
preserve(mlir::ModuleOp module,
         llvm::ArrayRef<MemRefContainerAccessContract> contracts);
} // namespace container::access::Contract

void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts);
void collectAsyncArgProvenanceContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<AsyncArgProvenanceContract> &contracts);
mlir::LogicalResult preserveLLVMAsyncArgProvenanceContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<AsyncArgProvenanceContract> contracts);

void collectNonObjectArgContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts);
void collectNonObjectArgContracts(
    mlir::ModuleOp module, const PyLLVMTypeConverter &typeConverter,
    llvm::SmallVectorImpl<NonObjectArgContract> &contracts);
mlir::LogicalResult preserveLLVMNonObjectArgContracts(
    mlir::ModuleOp module, llvm::ArrayRef<NonObjectArgContract> contracts);

void collectMemRefAtomicContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefAtomicContract> &contracts);
mlir::LogicalResult preserveLoweredMemRefAtomicContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefAtomicContract> contracts);

void collectMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefAggregateLoadContract> &contracts);
mlir::LogicalResult preserveLoweredMemRefAggregateLoadContracts(
    mlir::ModuleOp module,
    llvm::ArrayRef<MemRefAggregateLoadContract> contracts);

void collectMemRefDeallocContracts(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<MemRefDeallocContract> &contracts);
mlir::LogicalResult preserveLoweredMemRefDeallocContracts(
    mlir::ModuleOp module, llvm::ArrayRef<MemRefDeallocContract> contracts);

void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   LoweredSafetyContracts &contracts);
void collectLoweredSafetyContracts(mlir::ModuleOp module,
                                   const PyLLVMTypeConverter &typeConverter,
                                   LoweredSafetyContracts &contracts);
mlir::LogicalResult
preserveLoweredSafetyContracts(mlir::ModuleOp module,
                               const LoweredSafetyContracts &contracts);

class PyLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  explicit PyLLVMTypeConverter(mlir::MLIRContext *ctx,
                               mlir::ModuleOp module = {});

  mlir::Type getContainerStorageType(mlir::Type logicalType) const;
  mlir::MemRefType getListItemsMemRefType(ListType listType) const;
  mlir::Type getTupleItemsStorageType(TupleType tupleType) const;
  mlir::MemRefType getTupleItemsMemRefType(TupleType tupleType) const;
  mlir::MemRefType getDictKeysMemRefType(DictType dictType) const;
  mlir::MemRefType getDictValuesMemRefType(DictType dictType) const;

private:
  mlir::ModuleOp module;
};

namespace union_abi {

// A union value lowers to an i64 active-member tag followed by the concatenated
// parts of its non-None members in normalized member order. MemberSlice locates
// each member's payload parts inside the lowered value. The None member has an
// empty slice and is represented solely by its tag.
struct MemberSlice {
  mlir::Type memberType;
  unsigned offset = 0;
  unsigned count = 0;
};

std::optional<unsigned> memberTag(UnionType unionType, mlir::Type memberType);

mlir::LogicalResult
memberPartSlices(const PyLLVMTypeConverter &typeConverter, UnionType unionType,
                 llvm::SmallVectorImpl<MemberSlice> &slices);

} // namespace union_abi

class RuntimeAPI {
public:
  class Call {
  public:
    Call() = default;
    explicit Call(mlir::Operation *op) : op(op) {}

    mlir::Value getResult(unsigned index = 0) const {
      return op ? op->getResult(index) : mlir::Value();
    }

    mlir::Operation::result_range getResults() const {
      return op->getResults();
    }

    mlir::Operation *getOperation() const { return op; }
    mlir::Operation *operator->() const { return op; }
    explicit operator bool() const { return op != nullptr; }

  private:
    mlir::Operation *op = nullptr;
  };

  RuntimeAPI(mlir::ModuleOp module, mlir::OpBuilder &rewriter,
             const PyLLVMTypeConverter &typeConverter);

  Call call(mlir::Location loc, llvm::StringRef name,
            mlir::TypeRange resultTypes, mlir::ValueRange operands);
  Call call(mlir::Location loc, llvm::StringRef name, std::nullptr_t,
            mlir::ValueRange operands);
  Call call(mlir::Location loc, llvm::StringRef name, mlir::Type resultType,
            mlir::ValueRange operands);
  mlir::Value getByteLiteral(mlir::Location loc, mlir::StringAttr literal);
  mlir::Value getUnicodeLiteral(mlir::Location loc, mlir::StringAttr literal);
  mlir::Value getUnicodeLiteral(mlir::Location loc, mlir::StringAttr literal,
                                mlir::Type resultType);
  mlir::Value getNoneValue(mlir::Location loc);
  mlir::Value getI64Constant(mlir::Location loc, std::int64_t value);
  mlir::Value getF64Constant(mlir::Location loc, double value);
  mlir::OpBuilder &getBuilder() const { return rewriter; }

private:
  mlir::ModuleOp module;
  mlir::OpBuilder &rewriter;
};

// Pattern population functions

namespace lowering::value::base::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::base::Patterns

namespace lowering::try_ops::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::try_ops::Patterns

namespace lowering::value::list::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::list::Patterns

namespace lowering::value::number::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::number::Patterns

namespace lowering::value::class_::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::class_::Patterns

namespace lowering::value::tuple::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::tuple::Patterns

namespace lowering::value::dict::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::dict::Patterns

namespace lowering::value::union_::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::value::union_::Patterns

namespace lowering::call::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::call::Patterns

namespace lowering::refcount::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::refcount::Patterns

namespace lowering::func::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::func::Patterns

namespace lowering::func::definition::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::func::definition::Patterns

namespace lowering::func::returns::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::func::returns::Patterns

namespace lowering::func::objects::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::func::objects::Patterns

namespace lowering::async_runtime::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace lowering::async_runtime::Patterns

/// Optimization functions

namespace optimizer::publication {
/// Inserts explicit py.publish boundaries and computes publication summaries
/// before refcount insertion.
void prepare(mlir::ModuleOp module);
} // namespace optimizer::publication

namespace optimizer::pipeline {
/// Runs pre-lowering optimizations on Py dialect ops.
/// Call this after call conversion and before value conversion.
void preLowering(mlir::ModuleOp module);

/// Runs Python-value cleanup immediately after Py value conversion. This is
/// still the lowering boundary for object-level scalar concepts such as boxing
/// round trips and immortal singleton materialization.
void postValueLowering(mlir::ModuleOp module);

/// Runs only target-level cleanup after async/final LLVM conversion. High-level
/// concepts such as class layout, variables, ownership, and atomics must have
/// been optimized before this point.
void finalLLVMCleanup(mlir::ModuleOp module);
} // namespace optimizer::pipeline

/// Creates a pass that applies all Py-specific optimizations.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPyOptimizationPass();

/// Creates a pass that prepares explicit publication boundaries before
/// refcount insertion and runtime lowering.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPublicationPreparationPass();

/// Creates a pass that automatically inserts py.incref/py.decref operations.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass();

/// Creates a pass that removes only proven no-op py.incref/py.decref pairs.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass();

/// Creates a pass that verifies conservative quantitative ownership balance.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass();

/// Verifies the same conservative quantitative ownership balance used by
/// OwnershipVerifierPass. This is callable from lowering phases that mutate
/// ownership-carrying py.* ops before value conversion.
mlir::LogicalResult verifyOwnership(mlir::ModuleOp module);

/// Creates a pass that verifies lowered LLVM/runtime calls preserve ownership
/// balance for owned object-family descriptors.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass();

/// Creates a pass that verifies thread-safety contracts on memref/LLVM atomics.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMThreadSafetyVerifierPass();

/// Verifies ownership effects after py.incref/py.decref have been lowered to
/// LLVM/runtime calls. This catches post-lowering CSE or rewrite bugs that
/// duplicate, delete, or merge owned runtime call results.
mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module);

/// Creates a pass that verifies @native functions do not use py.* types.
/// This enforces the modal logic separation between Primitive World and Object
/// World.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createNativeVerificationPass();

} // namespace py
