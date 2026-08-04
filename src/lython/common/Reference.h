#pragma once

#include "Ownership.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace py::ownership {

// WHICH +1 A NAME DENOTES.
//
// An alias class is not a resource, and this is the distinction the ownership
// walks were missing. `AliasAnalysis` answers "do these two names reach the same
// entity"; every caller actually needs "do these two names denote the same
// REFERENCE", because one entity routinely carries several -- an owned call
// result and a token a retain minted on it are two increments with two releases
// and every name of one reaches the other.
//
// Answering it per call site produced sixteen predicates, four representations
// of a resource each caching the answer in its own fields, and a run of defects
// that could only be found one at a time: each site's approximation was wrong in
// a direction another site's approximation cancelled, and the suite checks the
// TOTAL number of releases rather than their attribution, so the sum matched
// while the attribution did not. Fixing one exposed the next
// (rfc/test-suite-debt.md).
//
// So the question is asked once, here, and the walks query the answer.
//
// ⛔ IT IS NOT THE ONLY QUESTION, and collapsing the sixteen into this one was
// measured and does not work. At least two more properties are orthogonal to it
// and load-bearing:
//
//   * whether a release is labelled `aggregate_release`, i.e. discharges the
//     CONTAINER's obligation rather than a local one -- dropping it double-frees
//     three `*_source_move_frequency` cases;
//   * whether the other reference was MINTED (`isMinted`) rather than received.
//
// A caller composes what it needs. This file answers one thing and nothing here
// knows which combination any walk wants.
//
// The model is three facts, and nothing in this file knows more than one of
// them:
//
//   * `referenceCreatorAt`   an operation increments, and one of its results
//                            names the increment
//   * `mintedReferenceOf`    a retain increments and the marker beside it is
//                            the only name the increment has
//   * `preservesReference`   an operation renames without incrementing
//   * `continuedReferenceOf`  a mutation hands back the reference it was given,
//                             under new lanes
//
// `ReferenceMap` is their transitive closure over SSA names, memoised.

// The operation that performed the increment, plus which of its increments.
// Null means "no claim": a borrowed name, or a shape this analysis does not
// model. Every consumer must read that as "could be anything", because it is
// the answer that makes a walk keep its pre-existing conservative behaviour.
struct Reference {
  mlir::Operation *creator = nullptr;
  unsigned index = 0;

  explicit operator bool() const { return creator != nullptr; }
  bool operator==(const Reference &other) const {
    return creator == other.creator && index == other.index;
  }
  bool operator!=(const Reference &other) const { return !(*this == other); }
};

// --- the three facts ---------------------------------------------------------

// Does `op` increment, with result #index naming the increment? True for an
// owned result of a call: the callee's contract says the caller receives a
// reference, and the result is its name.
//
// Not the retain primitive: it returns nothing, so no result of it can name what
// it created. That is `mintedReferenceOf`.
bool createsReferenceAtResult(FuncContractCache &contracts, mlir::Operation *op,
                              unsigned index);

// The retain whose increment `marker` is the only name of, or none.
//
// A `owned_local_object` marker is an identity cast, so its operands and its
// results reach the same entity -- but the results denote the retain's new
// increment and the operands denote whatever they denoted before. Walking
// through the cast (`underlyingObjectValue`) erases exactly that, which is why
// `ResourceGroup::root` is the same value for a token and for what it was minted
// on.
//
// A marker with no retain beside it REPUBLISHES: it adds no increment, so it
// denotes whatever its operand denotes and this returns none.
mlir::Operation *mintedReferenceOf(mlir::Operation *marker,
                                   AliasAnalysis &aliases);

// Does `op` rename without incrementing -- same entity, same reference, another
// spelling? Casts, prefix views, and the dialect's own identity ops.
//
// An `owned_local_object` marker is deliberately NOT one of these even though it
// is spelled as a cast: whether it renames or mints is `mintedReferenceOf`'s
// question, and answering it here would make this predicate depend on the alias
// relation, which it must not.
bool preservesReference(mlir::Operation *op);

// The operand whose reference `op`'s result #index continues, or none.
//
// A MUTATION CONTINUES A REFERENCE, it does not create one. `transfer_args`
// says the callee took the caller's, and an owned result handed back alongside
// is that same obligation with the payload re-rooted -- which is exactly what
// `advanceGroupLanesThroughReRoots` moves a group's lanes through. Reading the
// result as a new reference splits one obligation in two, and the walk then
// disowns its own release: measured on the three `*_source_move_frequency`
// cases, which place a second one.
//
// Ambiguous shapes answer none. More than one transferred argument, or an owned
// result at a position with no transferred argument to pair it with, is not a
// move this can name, and no claim is the conservative answer.
mlir::Value continuedReferenceOf(FuncContractCache &contracts,
                                 mlir::Operation *op, unsigned index);

// --- the closure -------------------------------------------------------------

class ReferenceMap {
public:
  ReferenceMap(FuncContractCache &contracts, AliasAnalysis &aliases)
      : contracts(contracts), aliases(aliases) {}

  // The reference `value` denotes, or none.
  Reference of(mlir::Value value) const;

  // Was this reference MINTED by a retain, as opposed to received from a call?
  //
  // The two are not interchangeable to a liveness walk. A minted reference is an
  // increment taken on top of whatever else holds the entity, so it keeps the
  // entity alive independently and uses under its names are safely somebody
  // else's; a received one may be the very reference that just died, and
  // dropping the pins under it releases early. Measured: two leak-gate members
  // regress the moment the liveness exclusion stops asking.
  bool isMinted(Reference reference) const;

  // Does `value` name an increment of its own, as opposed to borrowing one?
  // The predicate every walk needs before it may act on somebody else's release:
  // a group that borrows shares the reference and must keep reading its
  // discharge as its own.
  bool ownsReference(mlir::Value value) const {
    return static_cast<bool>(of(value));
  }

private:
  Reference compute(mlir::Value value) const;

  FuncContractCache &contracts;
  AliasAnalysis &aliases;
  mutable llvm::DenseMap<mlir::Value, Reference> cache;
};

} // namespace py::ownership

namespace llvm {
template <> struct DenseMapInfo<py::ownership::Reference> {
  using Reference = py::ownership::Reference;
  static Reference getEmptyKey() {
    return {DenseMapInfo<mlir::Operation *>::getEmptyKey(), 0};
  }
  static Reference getTombstoneKey() {
    return {DenseMapInfo<mlir::Operation *>::getTombstoneKey(), 0};
  }
  static unsigned getHashValue(const Reference &reference) {
    return hash_combine(reference.creator, reference.index);
  }
  static bool isEqual(const Reference &lhs, const Reference &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm
