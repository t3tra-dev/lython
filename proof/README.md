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
| `Proof.Memory.Lython` | the element signature Lython actually lowers to |
| `Proof.Memory.Trace`, `Proof.RC.Trace` | concrete traces, checked by computation |

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

## What is **not** proved, and should not be read as if it were

- **Nothing about the compiler.** These are theorems about the model. Connecting
  them to `src/lython/` is a refinement obligation that does not exist yet.
- **No concurrency.** No threads, no permissions, no data races. §5 and §6 of the
  note are untouched.
- **`WFRC` is never established.** The invariant is *stated* and the operations
  are proved to move both counts together, but no theorem yet says "every
  reachable machine satisfies `WFRC`". That is the preservation proof, and it is
  the next real piece of work in this layer.
- **No elaboration.** `Proof.QTT.Quantity` records that a quantity does not
  determine a reference mode, and stops there. Translating a `qω` source program
  into explicit `dup`/`drop` — the step where escape analysis and liveness come
  in — is not modelled.
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
