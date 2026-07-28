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

Steps 1–4 of the implementation order in the note: single memory space,
sequential execution, dynamic shape/offset/stride, the view-forming operations,
and alloc/dealloc with generations.

| module | content |
|---|---|
| `Proof.Prelude` | `Result`, byte ranges, the check combinators |
| `Proof.Memory.Fault` | the faults this model distinguishes, by name |
| `Proof.Memory.Byte` | `Byte`, and `uninit` as a first-class state |
| `Proof.Memory.Element` | `ElemSig`: element width, alignment, positivity |
| `Proof.Memory.Index` | in-bounds-by-construction multi-dimensional indices |
| `Proof.Memory.Heap` | allocation identity, blocks, byte read/write |
| `Proof.Memory.Descriptor` | strided MemRef descriptors, address arithmetic |
| `Proof.Memory.Resolve` | index + descriptor + heap → byte range, or a fault |
| `Proof.Memory.Ops` | alloc, load, store, dealloc, subview, view, casts |
| `Proof.Memory.Properties` | the theorems |
| `Proof.Memory.Lython` | the element signature Lython actually lowers to |
| `Proof.Memory.Trace` | a concrete trace, checked by computation |

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

## What is **not** proved, and should not be read as if it were

- **Nothing about the compiler.** These are theorems about the model. Connecting
  them to `src/lython/` is a refinement obligation that does not exist yet.
- **No concurrency.** No threads, no permissions, no data races. §5 and §6 of the
  note are untouched.
- **No reference counting.** `Own`/`Borrow`, owner sites, `logicalRC` — none of
  the second half of the note is here yet.
- **`realloc` is absent.** `Generation` exists and is checked, so the model is
  ready for it, but no operation bumps a generation, which means
  `stale-generation` is currently reachable only by constructing a descriptor by
  hand.
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

1. `realloc`, so `Generation` earns the check that already exists for it.
2. The QTT / reference-counting layer: `OwnerSite`, `logicalRC`, and the
   invariant that the runtime counter implements the number of owner sites.
3. Threads, spawn/join, and the permission algebra that makes race freedom
   provable.
4. A refinement from this model down to what `src/lython/lowering` emits. Until
   that exists, nothing here constrains the compiler.
