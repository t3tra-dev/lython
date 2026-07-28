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
| `Proof.RC.Object` | `ObjId`, `Life`, `RuntimeCount`, immortals |
| `Proof.RC.OwnerSite` | owner sites and `logicalRC` |
| `Proof.RC.Machine` | heap + object table + ghost site map |
| `Proof.RC.Ops` | **py.incref / py.decref**, move, borrow, reclaim |
| `Proof.RC.Invariant` | `WFRC`: what makes the counter mean something |
| `Proof.RC.Properties` | the refcount theorems |
| `Proof.Object.Word` | 8-byte words, encode/decode, and the bounded round trip |
| `Proof.Object.WordSig` | an element signature with a designated word type |
| `Proof.Object.Layout` | **the one-lane object layout**, and its disjointness facts |
| `Proof.Object.Box` | the box: one descriptor, fields as indices into it |
| `Proof.Object.Ops` | new / retain / release / move / free / resize |
| `Proof.Object.Coherence` | **alloc / free / move coherence** |
| `Proof.Program.Syntax` | the linear resource IR: names, blocks, terminators |
| `Proof.Program.Env` | names to entities, and **`Aliases`** |
| `Proof.Program.Step` | **the step relation**, unwind edges, reachability |
| `Proof.Program.Ownership` | ownership across a program |
| `Proof.Concurrent.Event` | events, `Conflict` |
| `Proof.Concurrent.Machine` | threads, scheduler, happens-before, `Race` |
| `Proof.Lython.Invalid` | **what Lython's semantics forbid** |
| `Proof.Lython.Detect` | the decision procedures, with soundness |
| `Proof.Memory.Lython` | the element signature Lython actually lowers to |
| `Proof.Memory.Trace`, `Proof.RC.Trace`, `Proof.Object.Trace`, `Proof.Program.Trace`, `Proof.Lython.Trace` | concrete traces, checked by computation |

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
- **No permission algebra, and so no race-freedom theorem.** `Race` and
  `RaceFree` are defined; nothing proves any program satisfies the second.
  **A predicate being definable is not the same as any program being shown free
  of it**, and the module says so in place.
- **`step-preserves-WFRC` is not exported.** Its five fields now rest on proved
  lemmas or on `reachable-keeps-the-heap`, and the two IR-level obstructions are
  gone — but each field is ∀-quantified over *all* objects, so every rule needs
  the untouched-object case too. That is assembly, not discovery, and it is not
  yet written. **"The obstruction is gone" and "the theorem is proved" are
  different statements** and only the first is claimed.
- **`WFRC` is never established.** The invariant is *stated* and the operations
  are proved to move both counts together, but no theorem yet says "every
  reachable machine satisfies `WFRC`". That is the preservation proof, and it is
  the next real piece of work in this layer.
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
- **`WellShaped` is a side proof.** A box is a bare `Desc 1`, so the shape
  invariant travels beside it rather than inside it. That is deliberate — a box
  that were a *pair* of descriptor and proof would be two values again — but it
  means nothing stops a caller pairing a well-formedness witness with the wrong
  descriptor. Making that impossible needs the witness erased at runtime and
  bundled at compile time, which Agda can express and this development does not
  yet do.
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

1. **Preservation of `WFRC`**: every operation takes a well-formed machine to a
   well-formed machine. Without it the invariant is a definition nobody has to
   satisfy.
2. **Elaboration**: `qω` source to a linear resource IR with explicit
   `dup`/`drop`, and the theorem that the result is well-typed.
3. Threads, spawn/join, and the permission algebra that makes race freedom
   provable.
4. A refinement from this model down to what `src/lython/lowering` emits. Until
   that exists, nothing here constrains the compiler.
