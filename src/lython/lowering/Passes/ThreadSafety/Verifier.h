#pragma once

#include "Common/AsyncSafetyKernel.h"
#include "Common/Container.h"
#include "Common/RuntimeSupport.h"
#include "Common/ThreadSafetyKernel.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::threadsafe {

namespace attrs {
std::optional<::llvm::StringRef> str(mlir::Operation *op,
                                     ::llvm::StringRef attrName);
std::optional<int64_t> i64(mlir::Operation *op, ::llvm::StringRef attrName);
mlir::SmallVector<int64_t> i64Array(mlir::Operation *op,
                                    ::llvm::StringRef attrName);
} // namespace attrs

namespace constant {
bool memrefInt(mlir::Value value, int64_t expected);
bool llvmInt(mlir::Value value, int64_t expected);
bool llvmBoolOrInt(mlir::Value value, int64_t expected);
bool llvmNullPtr(mlir::Value value);
std::optional<int64_t> anyInt(mlir::Value value);
std::optional<int64_t> index(mlir::Value value);
} // namespace constant

namespace header_slot {
std::optional<int64_t> refcount(mlir::Value header);
std::optional<int64_t> lock(mlir::Value header);
std::optional<int64_t> expectedRefcount(::llvm::StringRef kind);
std::optional<int64_t> expectedLock(::llvm::StringRef kind);
} // namespace header_slot

namespace memref_value {
bool alloca(mlir::Value value);
bool alloc(mlir::Value value);
} // namespace memref_value

namespace object_header {
bool type(mlir::Type type);
bool runtimeArg(mlir::BlockArgument arg);
bool provenance(mlir::Value value);
} // namespace object_header

namespace local_container {
bool escapeUser(mlir::Operation *op);
bool use(mlir::Operation *op, mlir::Value value);
mlir::Operation *escape(mlir::Value value,
                        ::llvm::SmallPtrSetImpl<mlir::Value> &seen);
} // namespace local_container

namespace provenance {
bool gep(mlir::Value pointer);
bool descriptorData(mlir::Value pointer);
bool descriptorAllocated(mlir::Value pointer);
bool entryArgRoot(mlir::Value value);
bool asyncExceptionCell(mlir::Value value);
bool asyncExceptionCellAllocated(mlir::Value value);
bool asyncCancelFlag(mlir::Value value);
bool asyncCancelFlag(mlir::Operation *op, mlir::Value value);
} // namespace provenance

namespace pointer {
mlir::Value stripCasts(mlir::Value value);
mlir::Value gepRoot(mlir::Value pointer);
} // namespace pointer

namespace descriptor {
std::optional<::llvm::StringRef> group(mlir::Value value);
std::optional<::llvm::StringRef> kind(mlir::Value value);
std::optional<::llvm::StringRef> component(mlir::Value value);

struct Kind {
  static std::optional<::llvm::StringRef> infer(mlir::Value header);
  static std::optional<::llvm::StringRef> get(mlir::Value header);
};

unsigned componentCount(mlir::Value header);
bool siblingIndex(mlir::Value header, mlir::Value component);
bool sameResource(mlir::Value header, mlir::Value component);
mlir::Value headerSibling(mlir::Value value);
bool value(mlir::Value value);
} // namespace descriptor

namespace resource {
std::string group(mlir::Value header);
void sealAtomic(mlir::Operation *op, mlir::Value header, int64_t slot);
void sealAccess(mlir::Operation *op, mlir::Value header, mlir::Value target);
} // namespace resource

namespace function_arg {
bool hasAttr(mlir::Value value, ::llvm::StringRef attrName);
bool loweredRank1MemRefDescriptor(mlir::BlockArgument arg,
                                  unsigned componentIndex);
} // namespace function_arg

namespace value_type {
bool llvmPointer(mlir::Type type);
} // namespace value_type

namespace atomic {
mlir::Value memrefHeader(mlir::Operation *op);
mlir::Value llvmPointer(mlir::Operation *op);
mlir::Value llvmRoot(mlir::Operation *op);
} // namespace atomic

namespace compare {
bool llvmZero(mlir::LLVM::ICmpOp cmp, mlir::Value value,
              mlir::LLVM::ICmpPredicate expected);
} // namespace compare

namespace retain_op {
bool runtimeCall(mlir::Operation *op);
bool atomic(mlir::Operation *op);
} // namespace retain_op

namespace dominance {
bool block(mlir::Block *block, mlir::Operation *op,
           mlir::DominanceInfo &dominance);
} // namespace dominance

namespace verifier::refcount {
struct RetainPremise {
  static mlir::LogicalResult verify(mlir::Operation *op);
};
} // namespace verifier::refcount

namespace verifier::memref {
struct AtomicRMW {
  static mlir::LogicalResult verify(mlir::memref::AtomicRMWOp op);
};
struct GenericAtomicRMW {
  static mlir::LogicalResult verify(mlir::memref::GenericAtomicRMWOp op);
};
struct Store {
  static mlir::LogicalResult verify(mlir::memref::StoreOp op);
};
struct Dealloc {
  static mlir::LogicalResult verify(mlir::memref::DeallocOp op);
};
struct Alloca {
  static mlir::LogicalResult verify(mlir::memref::AllocaOp op);
};
} // namespace verifier::memref

namespace verifier::container {
struct HeaderSlot {
  static mlir::LogicalResult verify(mlir::Operation *op,
                                    ::llvm::StringRef role);
};
struct BorrowRetain {
  static mlir::LogicalResult dominance(mlir::Operation *funcLike);
};
struct Access {
  static mlir::LogicalResult regions(mlir::Operation *funcLike);
  static mlir::LogicalResult coverage(mlir::Operation *funcLike);
  static mlir::LogicalResult final(mlir::Operation *funcLike);
};
struct DescriptorAccess {
  static mlir::LogicalResult final(mlir::Operation *funcLike);
};
} // namespace verifier::container

namespace control {
mlir::Value condition(mlir::Operation *terminator);
bool llvmReleaseToZero(mlir::Value condition, ::llvm::StringRef group);
bool noReturn(mlir::Operation *terminator);
} // namespace control

namespace verifier::llvm {
struct Ordering {
  static mlir::LogicalResult verify(mlir::LLVM::AtomicRMWOp op,
                                    mlir::LLVM::AtomicOrdering actual);
};
struct AtomicRMW {
  static mlir::LogicalResult verify(mlir::LLVM::AtomicRMWOp op);
};
struct RetainCall {
  static mlir::LogicalResult verify(mlir::LLVM::CallOp call);
};
struct LockAcquire {
  static mlir::LogicalResult controlFlow(mlir::LLVM::AtomicRMWOp op);
};
struct FreeCall {
  static mlir::LogicalResult verify(mlir::LLVM::CallOp call);
};
struct Store {
  static mlir::LogicalResult verify(mlir::LLVM::StoreOp op);
};
struct Load {
  static mlir::LogicalResult verify(mlir::LLVM::LoadOp op);
};
struct CmpXchg {
  static mlir::LogicalResult verify(mlir::LLVM::AtomicCmpXchgOp op);
};
} // namespace verifier::llvm

namespace verifier::async_runtime {
struct RefcountCall {
  static mlir::LogicalResult verify(mlir::Operation *op,
                                    mlir::ValueRange operands);
};
struct Cells {
  static mlir::LogicalResult verify(mlir::Operation *funcLike);
};
struct Handles {
  static mlir::LogicalResult balance(mlir::Operation *funcLike,
                                     mlir::Region &body);
};
} // namespace verifier::async_runtime

namespace verifier::class_helper {
struct Incref {
  static mlir::LogicalResult verify(mlir::LLVM::LLVMFuncOp fn);
};
struct Decref {
  static mlir::LogicalResult verify(mlir::LLVM::LLVMFuncOp fn);
};
struct DestroyLocal {
  static mlir::LogicalResult verify(mlir::LLVM::LLVMFuncOp fn);
};
struct Promote {
  static mlir::LogicalResult verify(mlir::LLVM::LLVMFuncOp fn);
};
} // namespace verifier::class_helper

namespace verifier::module {
mlir::LogicalResult verify(mlir::ModuleOp module);
} // namespace verifier::module

} // namespace py::threadsafe
