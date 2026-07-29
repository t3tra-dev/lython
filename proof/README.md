# `proof/` — the memory-safety kernel, in Agda

Lython's safety argument is rooted in a proof, not in the compiler's test suite.
This directory is where the proof lives. The intent is the order stated in
`.note/memsafe_proof.md`: **establish the safe operation mathematically here
first, then change the compiler to match it** — not the other way round.

```
make            typecheck everything reachable from src/Everything.agda
make redcheck   check the checker (see "Why redcheck exists" below)
make clean
```

Requires Agda ≥ 2.8 and agda-stdlib 2.x. The stdlib path is **discovered**, not
committed and not installed globally — see the head of the `Makefile` for why.
Override with `make AGDA_STDLIB=/path/to/standard-library.agda-lib`.

Everything is `--safe`. No `postulate`, no `{-# TERMINATING #-}`, no holes.

---

## What is modelled so far

Steps 1–4 of the note's implementation order, the memref dialect's memory API,
`realloc`, and the reference-counting layer.

| module | content |
|---|---|
| `Proof.Prelude` | `Result`, byte ranges, the check combinators |
| `Proof.Memory.Fault` | the faults this model distinguishes, by name |
| `Proof.Memory.Byte` | `Byte`, and `uninit` as a first-class state |
| `Proof.Memory.Element` | `ElemSig`: element width, alignment, positivity |
| `Proof.Memory.Index` | in-bounds-by-construction multi-dimensional indices |
| `Proof.Memory.Heap` | allocation identity, blocks, storage kind, frames |
| `Proof.Memory.Descriptor` | strided MemRef descriptors, address arithmetic |
| `Proof.Memory.Resolve` | index + descriptor + heap → byte range, or a fault |
| `Proof.Memory.Properties` | the memory theorems |
| `Proof.MemRef.Dialect` | **the memref op set, transcribed** |
| `Proof.MemRef.Realloc` | `realloc`, and the invalidation it forces |
| `Proof.QTT.Quantity` | multiplicities, `RefMode`, and what quantity does *not* decide |
| `Proof.QTT.Trace` | the erasure judgment, and the five refutations that give it content |
| `Proof.RC.Object` | `ObjId`, `Life`, `RuntimeCount`, immortals |
| `Proof.RC.OwnerSite` | owner sites and `logicalRC` |
| `Proof.RC.Machine` | heap + object table + ghost site map |
| `Proof.RC.Ops` | **py.incref / py.decref**, move, borrow, reclaim |
| `Proof.RC.Invariant` | `WFRC`: what makes the counter mean something |
| `Proof.RC.WellFormed` | **machines that satisfy `WFRC`**, at every point of the RC trace |
| `Proof.RC.Aggregate` | field paths, **multiplicity**, `aggregateRelease`, and the orphan |
| `Proof.RC.Properties` | the refcount theorems |
| `Proof.Object.Word` | 8-byte words, encode/decode, and the bounded round trip |
| `Proof.Object.WordSig` | an element signature with a designated word type |
| `Proof.Object.Layout` | **the one-lane object layout**, and its disjointness facts |
| `Proof.Object.Box` | the box: one descriptor, fields as indices into it |
| `Proof.Object.Ops` | new / retain / release / move / free / resize |
| `Proof.Object.Coherence` | **alloc / free / move coherence** |
| `Proof.Object.Shaped` | the shape witness bundled with its box, so it cannot be mis-paired |
| `Proof.Program.Syntax` | the linear resource IR: names, blocks, terminators |
| `Proof.Program.Env` | names to entities, and **`Aliases`** |
| `Proof.Program.Step` | **the step relation**, unwind edges, reachability |
| `Proof.Program.Ownership` | ownership across a program |
| `Proof.Program.Run` | **every instruction rule, taken as a step**, both counts read at each point |
| `Proof.Program.Coherence` | **owned names vs owner sites**: kept by the instructions, broken by `br`; **coherence = leak-freedom** |
| `Proof.Program.Preservation` | **⭐ every instruction step preserves the invariant** |
| `Proof.Concurrent.Event` | events, `Conflict` |
| `Proof.Concurrent.Machine` | threads, scheduler, happens-before, `Race` |
| `Proof.Concurrent.Trace` | **the scheduler taking steps**, a race, and three non-races |
| `Proof.Concurrent.RaceFree` | **⭐ race freedom**, without a permission algebra |
| `Proof.Lython.Invalid` | **what Lython's semantics forbid** |
| `Proof.Lython.Detect` | the four decision procedures, **sound and complete** |
| `Proof.Lython.Decide` | **⭐ `Valid`, decided by a FINITE check** over the state's own lists |
| `Proof.Memory.Lython` | the element signature Lython actually lowers to |
| `Proof.Memory.Trace`, `Proof.RC.Trace`, `Proof.Object.Trace`, `Proof.Program.Trace`, `Proof.Lython.Trace` | concrete traces, checked by computation |

Every predicate in the development is inhabited by one of the trace modules. That
is checked, not assumed — see "Why inhabitation is measured" below.

### Invalidity, matched to this language rather than in general

Deliberately **not** a permission algebra. A fractional-permission PCM answers
"may this thread touch these bytes" in general; what is needed is narrower — the
handful of things *this* language calls invalid, each stated so a program can be
shown not to reach it.

- **a refcount update on an object two threads can reach must be atomic.** The
  rule the GIL used to enforce and that PEP 703's biased counting replaces.
  Decidable from the site map, because owner sites are already thread-indexed —
  `field′`, `global` and `queue` sites belong to *no* thread, which is the
  escaped case and is where treating a reference as thread-local goes wrong.
- **⭐ and an immortal never needs one** — proved, not assumed. `{0,1,2}` are
  immortal in this runtime and shared by every thread that touches a small
  integer, so the atomics a conservative implementation emits there are
  *provably* unnecessary. This is the direction worth having: "we made it atomic
  everywhere" and "we proved where it need not be" are different results.
- **a container may not change length while an iterator over it is live.**
  A *semantic* rule, not a memory-safety one, and it survives one-laning
  untouched — which is why "we made it one lane" is not an answer to "does
  mutation during iteration still raise".
- **a borrow may not outlive its anchor.** `Mode.borrowed` records the name it
  came from, so the check is a second lookup and needs no analysis.
  `drop-strands-its-borrows` says *where* it belongs: at the drop, not at the
  borrow. **Sound *and* complete** — `danglingAnchor-exact`. Soundness alone is
  satisfied by a checker that reports nothing; completeness is what lets a pass
  conclude from silence, and `silence-means-safe` is that form.

### The one-lane object — a redesign, not a transcription

```
┌──────────────────── memref<N x i64> ────────────────────┐
│ rc │ class │ length │ capacity │ buf.alloc │ buf.gen │ … │
└─────────────────────────────────────────────────────────┘
  0     1        2         3           4          5      6…
```

**An object reference is ONE descriptor, and every field is an index into it.**
No `(header, payload)` pair, no side lane, no second SSA value travelling
alongside. Reading an object's class id is the same operation as reading its
third payload word, at a different number — there is no view, no
`reinterpret_cast`, and nothing to keep in sync.

The refcount is genuinely eight bytes *inside the object's own allocation*, not
a ghost field beside it. `Proof.Object.Word` proves `decode (encode n) ≡ n` for
`n < 256⁸`, and `Proof.Object.Trace` runs the encoder, the byte store, the byte
load and the decoder and checks the number that comes back. A ghost refcount
would make every one of those equations hold for a reason that has nothing to do
with the layout.

The mutable part lives in a **separate buffer** the box names by `(alloc, gen)`
in words 4 and 5 — the note's "stable box, resizable buffer" split. Resizing
reallocates the *buffer* and rewrites two words, so **every alias of the object
stays valid and every alias sees the new buffer**, because they all read word 4.
Reallocating the object itself would invalidate every alias, and there is no way
to reach them; `Proof.MemRef.Realloc.stale-descriptor-faults-after-realloc` is
that contrast as a theorem.

### The op set is Lython's, not the documentation's

Counted over `src/lython/runtime/modules/*.mlir` and the C++ builders:

```
store 668  load 658  cast 320  get_global 187  global 151  alloc 111
extract_aligned_pointer_as_index 84  dealloc 82  dim 58  generic_atomic_rmw 30
alloca 27  view 20  subview 16  + reinterpret_cast, extract_strided_metadata
```

**`memref.realloc` appears nowhere in this compiler**, and that is correct: a
boxed object is shared, realloc invalidates every alias, and nothing can update
the aliases it does not know about. It is modelled anyway, because `Generation`
exists to make that invalidation checkable and a generation nothing bumps is a
field no theorem can exercise.

Two ops carry warnings rather than just definitions.
`extract_aligned_pointer_as_index` (84 uses) is **where provenance is lost** —
its MLIR result is a bare `index` with no allocation, generation or liveness, so
a value obtained there and turned back into a pointer is outside this model
entirely. And the atomics deliberately have **no ordering parameter**, because
memref's do not; inventing one would describe a dialect that does not exist.

### The one design decision everything else follows from

> The heap is **byte storage with allocation provenance**; a MemRef is a
> **descriptor over it**; identity is `(allocation, generation, offset)` and
> never a physical address.

`Memory = Address → Byte` cannot express use-after-free at all: once the
allocator hands a freed address back, a stale pointer and a fresh one are the
same value, and no predicate over that heap can separate them.

---

## What is proved

In `Proof.Memory.Properties`, and unconditionally in `Proof.Memory.Trace`:

- **alloc is fresh** — it disturbs no existing allocation, and the descriptor it
  returns names the block it just made.
- **fresh storage is not readable** — a positive-width read of new memory is
  `uninitialized-read`, never a byte value. This is why `uninit` is a state and
  fresh blocks are not zero-filled.
- **use-after-free is caught, as use-after-free** — for every index and every
  rank, because the check that fails is on the block.
- **double-free is caught, as double-free** — a different constructor from
  use-after-free, so a compiler bug can be attributed to the pass that caused it.
- **a non-root descriptor cannot free** — a subview cannot free its parent,
  which is MLIR's "dealloc the memref that alloc returned" rule.
- **store/load round-trips, and only where it wrote** — the neighbouring element
  of a stored one is still `uninitialized-read`.
- **the check order in `resolve` is observable** — an i64 view at byte 1 of a
  4-byte block is `out-of-bounds` before it is `misaligned-access`.
- **an alloca and a global cannot be deallocated** — `invalid-free`, which is
  MLIR's rule and matters because this compiler emits 27 of the first and 151 of
  the second.
- **a descriptor that survives a `realloc` is refused as `stale-generation`** —
  every other check passes: the block is live and the right size. This is the
  one that generations exist for, and it is a *different* fault from
  use-after-free on purpose, because reporting the latter would send a reader
  looking for a missing `dealloc` that does not exist.

And in the reference-counting layer:

- **`py.incref` and `py.decref` move the runtime counter and the ghost owner
  count by the same amount** — stated as two equations that have to agree, not
  as one statement about the counter. A statement about only the counter is true
  of any implementation, including one that never counts.
- **`move` and `borrow` change neither** — this is the row of the table where
  the correct number of runtime operations is *zero*, and an elaborator that
  cannot see it emits a retain/release pair per read.
- **`reclaim` is refused while any site still holds the object** — so a
  premature free is a refusal at the counting layer rather than a
  use-after-free at the memory layer.
- **retaining into an occupied site, or releasing an unheld one, are refused** —
  each is a silent leak or a silent under-count otherwise.
- **immortals never reach zero**, by construction: there is no constructor of
  `ReachedZero` at `immortal`. Lython's small-int cache is exactly this, and one
  of its shipped defects turned on the `{0,1,2}` boundary.
- **⭐ `WFRC` is satisfiable, and is maintained across the whole trace** —
  `Proof.RC.WellFormed` carries a witness at each of allocate, incref, decref,
  decref-to-zero and reclaim. Every field is discharged with a *real* hypothesis
  somewhere; the module tabulates which, because a witness set in which some
  field is vacuous everywhere leaves that field possibly-unsatisfiable and every
  theorem resting on it worth what it was worth before.

And at the program layer, in `Proof.Program.Run`:

- **every instruction rule takes a step** — `new`, `dup`, `borrow`, `move`,
  `drop`, on one block, with each state written in *normal form* so the rule has
  to prove it produces what is written. Both counts are read at every point and
  `counts-agree` states the agreement as one equation per state.
- **⭐ owned names and owner sites are kept in step by every instruction, and
  broken by `br`** (`Proof.Program.Coherence`). That is what turns "the two
  counts disagree after a branch" from an observation into an *attribution*: the
  fix is not in either count, it is that `br` must become either a move or a dup.
- **⚠ two findings came out of running it.** `step-move` has no premise about
  names borrowed from its source, so a program can move out from under a live
  borrow — `move` is a second point, besides the drop, where the check has to go.
  And `reclaim`'s preconditions are met at a state where a *borrowed* name still
  denotes the object (`reclaim-is-licensed` typechecks): a borrow is a name that
  holds no site, so the ghost count reaching zero does not mean nothing refers to
  the object.

And the theorem the whole program layer was for, in
`Proof.Program.Preservation`:

- **⭐ every step takes a well-formed state to a well-formed state, and so does
  every run** — `reachable-preserves-WF`, over `—→*`: five instruction rules and
  five terminators, five invariant fields, quantified over *all* objects.
  `Proof.Program.Run` uses it: `WF s₆` is obtained from `WF s₀` by stepping, not
  by writing the witness down.
- **⭐ A BLOCK ARGUMENT IS A MOVE.** That was the open design decision and it is
  taken: `moveArgs` unbinds the operand and relocates its owner site to the
  parameter. The counter is untouched, so a block argument costs nothing — which
  is why move rather than dup: a loop-carried value would otherwise pay a
  retain/release pair per turn for a reference that never went anywhere. The
  operand's name is *gone* after the branch (`Proof.Program.Trace.x-is-gone`), so
  the two-owned-names state the shipped SIGSEGV turns on is not reachable.
- Two things had to change first, and both were changes to the IR rather than to
  the invariant. `no-stale-owner` moved from `strongAt` to **membership**,
  because `vacate` removes the first entry at a site and exposes the next, about
  which a first-entry property says nothing. And five rules gained premises —
  a `dup` of a dead object is a resurrection, a `new` onto absent storage is a
  dangling reference at birth, and `move`/`drop` on a *shadowed* name leave the
  environment owning a reference the site map no longer records. Each premise
  was mutation-tested: removing it makes the preservation proof fail.
- **The invariant is bigger than `WFRC` and has to be.** `backed` — every owned
  name occupies its own site holding its own entity — is what lets a rule that
  consults the *environment* be shown to do the right thing to the *machine*.

And in the concurrent layer, in `Proof.Concurrent.Trace`:

- **the scheduler takes steps**, in both positions of a two-thread pool, and a
  plain refcount RMW from each of two threads is a `Race` — with the
  happens-before refutation, not by assertion.
- **and three things are not races**: two atomics, two reads, and one thread with
  itself. Without those the definitions are consistent with `Conflict` holding of
  every pair.
- **⭐ race freedom for whole histories, without a permission algebra**
  (`Proof.Concurrent.RaceFree`). **Every access this IR performs is one word of
  one object's own allocation** — the one-lane layout leaves nothing else to
  point at — so two accesses overlap exactly when they name the same word of the
  same object, and the algebra a general model needs collapses into one
  arithmetic lemma (`aligned-blocks-disjoint`). What is left is two obligations
  belonging to different people: refcount traffic is the *compiler's* and is
  decided by `sharedPair`; payload traffic is the *program's* and no lowering can
  fix it. `history-is-race-free` discharges the first and takes the second as a
  hypothesis, which is the honest split.
- **⭐ different fields never conflict, and a field never touches the refcount
  word** — `HeaderWords` is positive, so payload traffic cannot be mistaken for
  refcount traffic. And **different objects never conflict**, which is the payoff
  of identity being provenance: with addresses two objects could share one and
  that theorem would be false.
- **A policy may emit *nothing*.** `Policy = ObjId → Maybe Atomicity`, and
  `nothing` is the right answer for an immortal — `bumpUp immortal ≡ immortal`,
  so there is nothing to write. `Proof.Concurrent.Trace` shows all three
  policies on one program: the naive one fails the obligation, atomic and
  eliding meet it.
- **⚠ `sched-step` was uninhabitable** and asking for an inhabitant is what found
  it. It carried the pool through unchanged and demanded the stepping thread be
  *identical* afterwards — which no real step can satisfy — and it left the
  recorded event unconstrained, so a single-threaded program could fabricate a
  race. Both are fixed.

And for the one-lane object:

- **one lane, with content** — any two field accesses of the same object land in
  the *same block*, because both resolutions report the block that one lookup of
  one allocation returned (`resolveIn-block`, `two-fields-one-block`). This is
  the statement a `(header, payload)` pair cannot make: there the two accesses
  consult two allocations, and once one is reallocated nothing relates them.
- **free is guarded** — a nonzero refcount refuses, and the freed lane then
  faults. The reference the program was holding *is* the root descriptor, so
  `dealloc` accepts it directly; a two-lane design has to reconstruct a
  descriptor nothing held, which is where a wrong width becomes a wrong
  deallocator.
- **move is free** — the moved reference reads the same refcount, the same class
  and the same heap. That is what makes eliding a retain/release pair around a
  move *correct* rather than merely cheaper.
- **the header survives** — a payload write does not touch the refcount or the
  class, and a refcount write does not touch the class. Checked at concrete
  values (`42 ≠ 1`), not only asserted by the disjointness lemmas.
- **reallocating the buffer leaves the box bit for bit** — `lookup-update-other`
  applied to the two distinct allocations, which is the payoff of the split.

## What is **not** proved, and should not be read as if it were

- **Nothing about the compiler.** These are theorems about the model. Connecting
  them to `src/lython/` is a refinement obligation that does not exist yet.
- **Race freedom is for a PAIR of emissions, not for a whole history.** Lifting
  it needs every event in a history to be shown to come from `eventFor`, which
  is an induction over `⇒*`. Mechanical, and not written.
- **`PayloadSeparated` is a hypothesis, deliberately.** Two threads writing one
  field of one object race in Lython exactly as in CPython without the GIL, and
  no lowering can prevent it. What a lowering *can* do is not add traffic of its
  own, and that half is decided.
- **`FollowsTheChecker` is still ∀-quantified over `ObjId`.** The four
  invalidity checks are not — `Proof.Lython.Decide` runs them over the lists the
  state provides and `Proof.Program.Coherence` establishes `Valid` on a
  reachable state that way. The emission-side obligation has no such list
  because it is about instructions the compiler is *about to write*, not about
  a state.
- **`Aggregate` does not distinguish "unset" from "absent".** It is a path
  through the site map, so a field the class declares but nothing has set has no
  entry and no path. Right for release; not enough for a reachability analysis.
  Distinguishing them needs the field slots in the cell, not just their count.
- **No elaboration.** `Proof.QTT.Quantity` records that a quantity does not
  determine a reference mode, and stops there. Translating a `qω` source program
  into explicit `dup`/`drop` — the step where escape analysis and liveness come
  in — is not modelled.
- **The object layer loses one attribution.** A second `freeObject` reports
  `use-after-free`, not `double-free`: it reads the refcount *through the lane*
  before deciding, so on a freed object the read faults first. The memory layer
  still tells the two apart; the object layer cannot, because its precondition
  lives in the storage it is about to release. Consulting the count some other
  way would mean a second lane — which is the thing this design removes. It is a
  real trade and `Proof.Object.Trace.double-free-reports-use-after-free` is what
  it costs.
- **`WellShaped` is bundled but not erased.** `Proof.Object.Shaped` makes the
  wrong pairing impossible — the witness travels inside the value and
  `allocShaped` builds it by `refl`, so nobody supplies one. Run-time erasure is
  a different matter and Agda refuses it: `@0` forbids transporting a *relevant*
  value along an erased equation, and `wordIx`'s result type `Ix (sizes b)`
  needs `spans-box`. Genuine erasure needs the index type to stop depending on
  the proof, which is a redesign of `Proof.Object.Box`.
- **No finalizers, no weak references, no cycles.** `Life` has a `finalizing`
  state and `reclaim` requires it, but nothing runs user code in that window.
- **The root token is weaker than the note asks for.** The note wants a linear
  token that `alloc` mints and `dealloc` consumes. Agda is not linear, so a token
  record would be constructible by anyone and would prove nothing. `IsRootOf` is
  a decidable fact about the descriptor and the block instead. It rejects freeing
  through a view; it does **not** make double-free unrepresentable, which is why
  double-free is a runtime fault here rather than a type error.

---

## Why `redcheck` exists

`make` passing means Agda accepted the development. It does not by itself mean
Agda would have *rejected* a wrong one — a misconfigured checker accepts
everything, and a broken harness fails everything.

So `make redcheck` runs two sentinels: a true lemma that must be **accepted**,
and a false one that must be **rejected**. Both directions are required, and the
positive one is not decoration. The first version of that target had a scope
error in its sentinel, so Agda rejected it for a reason unrelated to the lemma
being false — and the target printed "the checker is live". A guard whose pass
criterion is "the tool failed" passes whenever the guard itself is broken.

## Why inhabitation is measured

Agda tells you a theorem is correct. It does not tell you the theorem has
*content*. `WFRC m → X` is unconditionally true when nothing satisfies `WFRC`,
and `¬ P` is free when `P` is empty — so a development can be entirely `--safe`,
mutation-tested, and still be about nothing.

The only instrument for that layer is an **inhabitation census**: for each
predicate, is one ever built in a trace? Running it here found that `WFRC` had
never been satisfied, that no instruction rule had ever been applied, and that
the whole concurrent layer had never taken a step — through a rule that turned
out to be unsatisfiable.

The census itself failed twice before it worked, both times returning "0 for
everything". That is the dangerous shape: **the warning condition and the
instrument's own failure mode coincide exactly.** Verify the census against a
known inhabitant before believing any zero it reports.

## Why the trace module exists

Every theorem in `Properties` is conditional: *if* the lookup succeeds *and* the
generation matches *and* the block is live, *then* …. A model in which no heap
ever satisfies those hypotheses satisfies all of them and describes nothing.

`Proof.Memory.Trace` is unconditional equations about actual heaps, and each
`refl` is the typechecker running the model and agreeing. It has already earned
its place: it rejected an assertion of mine about which fault a misaligned
out-of-bounds view produces.

The same reasoning applies to `Proof.Memory.Lython`. The modules above it are
parameterised over an element signature, and a parameter that is never supplied
makes every theorem below it vacuous — this project already has a name for that
failure, `NonInstantiationIsNotConformance` in `rfc/memory-safety-proof.md`.

---

## Next

In the note's order (§9), and each is a separate piece of work:

1. **Decide what `br` is.** `step-preserves-WF` covers every instruction and
   stops at argument passing. The IR has to make a block argument either a MOVE
   (the operand's name dies at the branch) or a DUP (it occupies a site). That
   is a design decision this development has localised and not taken.
2. **Elaboration**: `qω` source to a linear resource IR with explicit
   `dup`/`drop`, and the theorem that the result is well-typed.
3. **Field footprints and the permission algebra** they need — the one place a
   general PCM is actually required.
4. A refinement from this model down to what `src/lython/lowering` emits. Until
   that exists, nothing here constrains the compiler. All four checkers are
   sound *and* complete and each rests on exactly one representational decision
   in the compiler; `danglingAnchor` (the anchor recorded in `Mode.borrowed`) is
   still the cheapest to transfer.
