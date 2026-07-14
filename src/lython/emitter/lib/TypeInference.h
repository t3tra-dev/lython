#pragma once

#include "PyDialectTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace lython::parser {
struct Node;
} // namespace lython::parser

namespace lython::emitter {

// Algorithm J inference core: a union-find store over py::InferVarType ids
// with Rémy-level generalization. The MLIR type tree stays immutable — all
// mutable unification state lives here, keyed by variable id, so the engine
// can hand plain mlir::Type values to the manifest/subtype machinery without
// a mirror type algebra.
class InferenceContext {
public:
  // Inference variables are engine-owned unknowns (unannotated parameters,
  // empty literals) and unify strictly. Instantiation variables come from
  // alpha-renaming a scheme at a use site; only they keep the legacy
  // join-relaxation at call-argument positions (bindTypeParameter's
  // bidirectional typeAccepts), because typeshed-style generic calls such as
  // max(1, 2.5) rely on it.
  enum class VarKind { Inference, Instantiation };

  struct UnifyResult {
    bool ok = true;
    std::string reason;

    explicit operator bool() const { return ok; }
  };

  explicit InferenceContext(mlir::MLIRContext &context) : context(context) {}

  mlir::MLIRContext &getContext() const { return context; }

  mlir::Type freshVar(VarKind kind, const parser::Node *origin = nullptr,
                      llvm::StringRef role = {});

  VarKind varKind(unsigned id) { return vars[findRep(id)].kind; }
  const parser::Node *varOrigin(unsigned id) { return vars[findRep(id)].origin; }
  llvm::StringRef varRole(unsigned id) { return vars[findRep(id)].role; }

  // Follows var -> binding chains until a non-variable type or an unbound
  // variable (returned as its canonical representative).
  mlir::Type resolveShallow(mlir::Type type);
  // Deep resolution over the whole type tree. Unbound variables remain (as
  // their canonical representatives) — callers decide whether that is a
  // diagnostic (facade boundary) or fine (intermediate step).
  mlir::Type zonk(mlir::Type type);
  bool fullyResolved(mlir::Type type) {
    return !py::containsPyInferVar(zonk(type));
  }

  // Equational unification with occurs check. Strict on ground mismatches:
  // subsumption (manifest subtyping, literal widening) is a separate,
  // directed layer on top — folding it in here would make inferred types
  // dependent on unification order.
  UnifyResult unify(mlir::Type a, mlir::Type b);

  // Replaces unbound variables deeper than baseLevel by fresh universally
  // quantified TypeVarTypes ($J0, $J1, ...). The result is an ordinary
  // TypeVar-bearing type, so the existing scheme consumers (call-site
  // binding, manifest serialization) work unchanged.
  mlir::Type generalize(int baseLevel, mlir::Type type);

  // Alpha-renames quantified type parameters to fresh Instantiation
  // variables. ParamSpec/TypeVarTuple packs are left untouched: they stand
  // for parameter lists, not single types, and keep flowing through the
  // existing TypeBindingMap expansion until that machinery is subsumed.
  mlir::Type instantiate(mlir::Type scheme, const parser::Node *origin = nullptr);

  int currentLevel() const { return level; }

  // Monotonically increasing count of successful bindings and variable
  // merges. Fixpoint drivers compare generations across a sweep instead of
  // tracking which unification produced progress.
  unsigned generation() const { return generationCounter; }

  class LevelScope {
  public:
    explicit LevelScope(InferenceContext &owner) : owner(owner) {
      ++owner.level;
    }
    LevelScope(const LevelScope &) = delete;
    LevelScope &operator=(const LevelScope &) = delete;
    ~LevelScope() { --owner.level; }

  private:
    InferenceContext &owner;
  };

  // Reserved quantifier prefix for generalization output. Manifest-declared
  // parameters use plain user spellings (T, $T contracts); the J prefix
  // guarantees generalized schemes never collide with them.
  static constexpr llvm::StringLiteral generalizedPrefix{"$J"};

  // Candidate exploration (overload selection, union-member trials) must not
  // leave bindings from rejected candidates in the store. While at least one
  // Speculation is live, every variable mutation is journaled; destruction
  // without commit() restores the journal back to the construction mark.
  // Commit keeps the entries so an enclosing speculation can still roll them
  // back. Fresh variables created under a rolled-back speculation survive as
  // unreferenced orphans on purpose — reclaiming them would invalidate ids.
  class Speculation {
  public:
    explicit Speculation(InferenceContext &owner)
        : owner(owner), mark(owner.trail.size()) {
      ++owner.speculationDepth;
    }
    Speculation(const Speculation &) = delete;
    Speculation &operator=(const Speculation &) = delete;
    ~Speculation() {
      if (!committed)
        owner.rollbackTo(mark);
      if (--owner.speculationDepth == 0)
        owner.trail.clear();
    }
    void commit() { committed = true; }

  private:
    InferenceContext &owner;
    std::size_t mark;
    bool committed = false;
  };

private:
  struct VarInfo {
    unsigned parent;
    unsigned rank = 0;
    mlir::Type binding;
    int level;
    VarKind kind;
    const parser::Node *origin;
    std::string role;
  };

  unsigned findRep(unsigned id);
  void recordMutation(unsigned id) {
    if (speculationDepth != 0)
      trail.push_back(TrailEntry{id, vars[id]});
  }
  void rollbackTo(std::size_t mark) {
    while (trail.size() > mark) {
      vars[trail.back().id] = std::move(trail.back().saved);
      trail.pop_back();
    }
  }
  // Occurs check and Rémy level adjustment share one traversal: both must
  // visit exactly the free variables of the bound type, and a second walk
  // would re-resolve the same chains.
  bool occursAndAdjustLevels(unsigned rep, int varLevel, mlir::Type type);
  UnifyResult bindVar(unsigned rep, mlir::Type type);
  UnifyResult unifyLists(mlir::ArrayRef<mlir::Type> as,
                         mlir::ArrayRef<mlir::Type> bs, mlir::Type a,
                         mlir::Type b);
  UnifyResult mismatch(mlir::Type a, mlir::Type b);

  struct TrailEntry {
    unsigned id;
    VarInfo saved;
  };

  mlir::MLIRContext &context;
  std::vector<VarInfo> vars;
  std::vector<TrailEntry> trail;
  unsigned speculationDepth = 0;
  int level = 0;
  unsigned generalizedCounter = 0;
  unsigned generationCounter = 0;
};

} // namespace lython::emitter
