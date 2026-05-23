#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"

namespace py {

namespace ordering {

inline bool atLeastAcquire(mlir::LLVM::AtomicOrdering ordering) {
  return ordering == mlir::LLVM::AtomicOrdering::acquire ||
         ordering == mlir::LLVM::AtomicOrdering::acq_rel ||
         ordering == mlir::LLVM::AtomicOrdering::seq_cst;
}

inline bool atLeastRelease(mlir::LLVM::AtomicOrdering ordering) {
  return ordering == mlir::LLVM::AtomicOrdering::release ||
         ordering == mlir::LLVM::AtomicOrdering::acq_rel ||
         ordering == mlir::LLVM::AtomicOrdering::seq_cst;
}

inline bool atLeastAcqRel(mlir::LLVM::AtomicOrdering ordering) {
  return ordering == mlir::LLVM::AtomicOrdering::acq_rel ||
         ordering == mlir::LLVM::AtomicOrdering::seq_cst;
}

inline bool refcountInc(mlir::LLVM::AtomicOrdering ordering) {
  return ordering == mlir::LLVM::AtomicOrdering::monotonic ||
         atLeastAcquire(ordering);
}

inline bool atLeastAcquire(llvm::StringRef ordering) {
  return ordering == ThreadSafetyAttrs::kOrderingAcquire ||
         ordering == ThreadSafetyAttrs::kOrderingAcqRel ||
         ordering == ThreadSafetyAttrs::kOrderingSeqCst;
}

inline bool atLeastRelease(llvm::StringRef ordering) {
  return ordering == ThreadSafetyAttrs::kOrderingRelease ||
         ordering == ThreadSafetyAttrs::kOrderingAcqRel ||
         ordering == ThreadSafetyAttrs::kOrderingSeqCst;
}

inline bool atLeastAcqRel(llvm::StringRef ordering) {
  return ordering == ThreadSafetyAttrs::kOrderingAcqRel ||
         ordering == ThreadSafetyAttrs::kOrderingSeqCst;
}

inline bool refcountInc(llvm::StringRef ordering) {
  return ordering == ThreadSafetyAttrs::kOrderingMonotonic ||
         atLeastAcquire(ordering);
}

inline bool atLeastAcquire(llvm::AtomicOrdering ordering) {
  return ordering == llvm::AtomicOrdering::Acquire ||
         ordering == llvm::AtomicOrdering::AcquireRelease ||
         ordering == llvm::AtomicOrdering::SequentiallyConsistent;
}

inline bool atLeastRelease(llvm::AtomicOrdering ordering) {
  return ordering == llvm::AtomicOrdering::Release ||
         ordering == llvm::AtomicOrdering::AcquireRelease ||
         ordering == llvm::AtomicOrdering::SequentiallyConsistent;
}

inline bool atLeastAcqRel(llvm::AtomicOrdering ordering) {
  return ordering == llvm::AtomicOrdering::AcquireRelease ||
         ordering == llvm::AtomicOrdering::SequentiallyConsistent;
}

inline bool refcountInc(llvm::AtomicOrdering ordering) {
  return ordering == llvm::AtomicOrdering::Monotonic ||
         atLeastAcquire(ordering);
}

} // namespace ordering

namespace role {

inline bool retainRefcount(llvm::StringRef role) {
  return role == ThreadSafetyAttrs::kRoleContainerRefcountRetain ||
         role == ThreadSafetyAttrs::kRoleClassRefcountRetain;
}

inline bool releaseRefcount(llvm::StringRef role) {
  return role == ThreadSafetyAttrs::kRoleContainerRefcountRelease ||
         role == ThreadSafetyAttrs::kRoleClassRefcountRelease;
}

inline bool lockAcquire(llvm::StringRef role) {
  return role == ThreadSafetyAttrs::kRoleContainerLockAcquire ||
         role == ThreadSafetyAttrs::kRoleClassLockAcquire;
}

inline bool lockRelease(llvm::StringRef role) {
  return role == ThreadSafetyAttrs::kRoleContainerLockRelease ||
         role == ThreadSafetyAttrs::kRoleClassLockRelease;
}

inline bool containerAtomic(llvm::StringRef role) {
  return role == ThreadSafetyAttrs::kRoleContainerRefcountLoad ||
         role == ThreadSafetyAttrs::kRoleContainerRefcountRetain ||
         role == ThreadSafetyAttrs::kRoleContainerRefcountRelease ||
         role == ThreadSafetyAttrs::kRoleContainerLockAcquire ||
         role == ThreadSafetyAttrs::kRoleContainerLockRelease;
}

} // namespace role

} // namespace py
