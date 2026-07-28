# What the model excludes, and what it cannot yet say

Reconciling `proof/` against the memory-safety defects actually found in
`src/lython/`. Every compiler-side fact below was measured in-tree; every
model-side fact was checked against the Agda rather than remembered.

The question is not "is the model good". It is **which of the defects this
compiler has actually shipped could this model have ruled out**, and for the
rest, **what vocabulary is missing**.

Four verdicts are used, and the difference between the last two is the whole
point of the document:

| verdict | meaning |
|---|---|
| **excluded** | the model makes the defect unrepresentable |
| **proved** | statable, and there is a theorem |
| **statable** | the vocabulary exists, no theorem yet |
| **inexpressible** | the model has no words for it |

---

## 1. The defects, against the model

### Excluded by construction — 1

| defect | why it cannot happen here |
|---|---|
| **stale lane after growth** (`before = u; u.update(big)` answered `False`/`0` where CPython says `True`/`203`, exit 0, SIGSEGV under libgmalloc) | The object is **one descriptor**. There is no second lane for a holder to keep, so the shape has nowhere to live. `two-fields-one-block` is the positive form: any two field accesses land in the same block. |

This is the one place the redesign earns its keep outright, and it is worth
being precise about how much: it excludes the *representation* that made the
defect possible, not the class of reasoning error that produced it.

### Proved — 5

| defect | theorem |
|---|---|
| use-after-free read as a value | `use-after-free-is-caught` |
| double-free read as use-after-free | `double-free-is-caught` (distinct constructors) |
| freeing through a view | `non-root-cannot-free` |
| freeing an `alloca` or a global (this compiler emits 27 and 151) | `alloca-cannot-be-deallocated`, `global-cannot-be-deallocated` |
| a reference surviving a `realloc` | `stale-descriptor-faults-after-realloc` |

### Statable but unproved — 2

| gap | why it is not yet a theorem |
|---|---|
| `WFRC` holds of every reachable machine | there is **no notion of "reachable"**: no step relation, so "every reachable machine" has no referent |
| the runtime counter never drifts from the owner count over a program | same reason — the per-operation lemmas exist (`retain-ghost`, `retain-counter`), nothing composes them |

### Inexpressible — 11, and they are the majority

| defect (all measured on this tree) | what is missing |
|---|---|
| **`for` in `try` SIGSEGV** — `rc=139`, deterministic, on `try: for v in [1,2]: total += v except: total += 100`. Root: `for` allocates the cell once and threads it as a block argument, so **one allocation has two SSA names and the ownership machinery makes that two entities**; the unwind pad frees the cell before the handler in the same function reads it | CFG, unwind edges, **SSA identity** |
| **normal-path double dealloc** — the shipped compiler runs `__ly_dealloc___ly_cell_1` twice for the same cell; benign only because the freed read does not come back as 1 | CFG |
| **loop `else`-only write lost** — `in else 7 / after 0`, silent, rc=0 | CFG |
| **family C**: release placement abandoned when one consuming use existed (`consumeIsDeath=false`) — 6 goldens green while leaking | CFG + placement |
| **families A/B**: latent double release on unwind edges no input reaches (`stdlib_string_catch` −93 DecRefs) | CFG + unwind edges |
| **family D**: retain omitted by provenance, not layout | SSA provenance (`func.call` result vs `memref.alloc` result) |
| **family E**: block-argument index space vs successor-operand index space | CFG, block arguments |
| **sequence/dict literal source move** — a use-**set** fact standing in for an execution-**frequency** fact; nondeterministically silent-wrong or aborting | **execution frequency**: the model has no loops and no notion of how often an operation runs |
| **read-back token** — `retainEvidenceElement` mints a second owned token on the same SSA values, and the placer finds the literal's `aggregate_release` among the alias equivalents | SSA aliasing + **aggregates** |
| **holder discharge / remaining leak families** — need a token **count**, not a token **name** | aggregates with multiplicity |
| **deallocator selection**: 5 of 14 widths shared, `dict` in a 7-way tie, `N` user classes giving `N+1` candidates at `memref<16xi64>` | **contracts**. The model has exactly one way to free a block; the compiler's problem is *choosing which deallocator*, which presupposes a manifest |

---

## 2. The five structural gaps

### 2.1 There is no program

The model has **operations** and no **program**. `Proof.Memory.Properties` and
`Proof.RC.Properties` prove things about one operation applied to one state.
There is no step relation, no basic block, no unwind edge, no trace — checked,
not assumed: nothing in `src/` matches `Step`, `—[`, `BasicBlock` or an event
relation.

The note asks for this in §5 and it was skipped in favour of getting the state
model right first. That was defensible; it is now the binding constraint.

**Cost, exactly.** Six of the eleven inexpressible defects are *placement*
defects — the operation is right and the position is wrong. A model with no
positions cannot have a wrong one. `WFRC` preservation is unprovable for the
same reason: "reachable" is undefined.

### 2.2 There is no SSA layer, and this is a level mismatch rather than a gap

This is the sharpest finding, and it is not "add a feature".

> The model is about **allocations and objects**.
> The compiler's defects are about **SSA values and control flow**.
> They do not meet, and the missing piece is a layer in between.

The clearest case is the shipped SIGSEGV. Its root is:

> one allocation, **two SSA names**, and the ownership machinery treats that as
> two entities.

There is no sentence in the current model that says this, because there is no
notion of a *name* distinct from an *allocation*. `AllocId` is the identity; a
descriptor is a value; nothing maps names to entities. The same is true of
family D (retain decided by *which op produced the value*) and of the read-back
token (a second token minted **on the same SSA values**).

The note describes the missing layer in §4: a **linear resource IR** with
explicit `dup`/`drop`/`move`/`borrow`, typed with separated owned and borrowed
contexts. `Proof.QTT.Quantity` records the vocabulary and stops. Until that
layer exists, `proof/` cannot be about the same objects the compiler's passes
manipulate, and any "refinement to `src/lython`" would have to bridge two levels
at once.

### 2.3 A leak cannot be stated

`WFRC.owned-storage-live` reads

```agda
owned-storage-live : ∀ o → 0 < ghostRC m o → … liveness b ≡ blockLive
```

— **owned ⇒ live**. The converse, *live ⇒ owned*, is what a leak violates, and
it is absent.

This is not academic. The suite's `leak` stage exists because at least seven
goldens were green while leaking, one defect leaked 64 bytes per iteration
without bound, and two leak families remain open. **The proof layer currently
cannot express the property that gate was built to check.**

Adding the converse is not free: it is false during construction, when a fresh
object is live and not yet stored anywhere. It needs the `finalizing` window and
a notion of a transient owner site — the `temp` constructor of `OwnerSite`
exists for exactly this and nothing yet uses it.

### 2.4 There is no contract, so deallocator selection is invisible

In the model, `dealloc` takes the descriptor and there is one way to free it. In
the compiler there are 34 deallocators, selection is by shape and arity and
declared name, **5 of 14 single-input widths are shared**, and `N` user classes
produce `N+1` candidates at width 16 because a per-class deallocator is
synthesised with no shape.

None of that has a counterpart here — no `ly.runtime.contract`, no manifest, no
candidate set, no tie. The measurements that closed the width question
(`tiecensus.py`, `preemption.py`, `laneswap.py`) are all compiler-side and have
no model-side statement to be checked against.

Whether that is a gap or a *correct simplification* is a real question, and the
honest answer is: **the redesign removes the need for selection** — if an object
reference is the root descriptor of its own allocation, freeing needs no
candidate set. So this may be a gap that closes by construction rather than by
addition. That has not been demonstrated, and stating it here is a hypothesis,
not a result.

### 2.5 The box invariant travels beside the box

`WellShaped` is a side proof. Nothing stops a caller pairing a witness with the
wrong descriptor. Deliberate — a box that were a *pair* of descriptor and proof
would be two values again, which is what one-laning removes — but not free.

The fix Agda supports is an erased field: bundle the witness at compile time and
erase it at runtime, so the box is still one value in the extracted code. Not
done.

---

## 3. What the model claims that the compiler does not satisfy

Worth separating from the gaps: these are places where the model is **stricter**
than the implementation, and closing them is compiler work rather than proof
work.

| model | compiler today |
|---|---|
| an object reference is the root descriptor of its own allocation | **four remain multi-value**: `int`, `str`, the exception family and class instances still carry interior state as a tuple of SSA values beside the root (`rfc/memory-safety-proof.md`). Every container and every fixed-width scalar is already one lane |
| fresh storage reads as `uninitialized-read` | no uninitialised-read check exists at any layer |
| `dealloc` refuses non-root descriptors | enforced by shape matching, which was measured to be ambiguous in 21–98 places per program |
| the refcount is inside the object's own allocation | true today |
| resizing reallocates the **buffer**, never the object | true today — `memref.realloc` appears nowhere, and the box/payload split is already the shape |

The last two are the encouraging rows: the redesign is not a departure from what
the compiler does, it is a *completion* of a split the compiler already made.

---

## 4. Barriers, in dependency order

1. **A step relation.** Nothing else on this list can be built without it.
   Sequential, finite traces, reflexive-transitive closure — the note's §5 is
   explicit that coinduction is not needed for safety. Unlocks: `WFRC`
   preservation, every placement defect, "reachable".

2. **A linear resource IR with SSA names and a CFG.** The layer the note
   describes in §4. This is where the compiler's actual passes live, and until
   it exists the proof and the implementation are about different things.
   Unlocks: the SIGSEGV class, families A–E, the frequency defects.

3. **Aggregates with multiplicity.** `aggregate(parent, path)` as a judgment,
   not a comment. The two remaining leak families were characterised as needing
   a token *count* rather than a token *name*, which is exactly what this is.

4. **The leak direction of the invariant.** live ⇒ owned, with the construction
   window handled by `temp` sites.

5. **Erased `WellShaped`.** Small, and it removes the one way the box invariant
   can be misapplied.

6. **A refinement to the emitted IR.** Last, and only meaningful once (2) exists
   — otherwise it spans two levels at once.

Concurrency (§5–§6 of the note: threads, permissions, race freedom) is
deliberately not on this list. It is a large piece of work and **not** on the
path to any defect this compiler has actually shipped, all of which are
single-threaded.

---

## 5. Honest summary

Of the seventeen memory-safety defects this session found and measured:

- **1** is excluded by the redesign outright
- **5** have theorems in the model
- **11** are inexpressible, and **6 of those 11 are placement defects** that need
  a control-flow layer the model does not have

The model is a good *state* model and not yet a model of a *program*. That is
the gap, stated in one line, and every item in §4 is downstream of it.

Two things should not be read into the current state. The proof directory does
**not** yet constrain the compiler — no refinement exists, and the README says
so. And the one-lane object is a redesign whose value is demonstrated against
one defect class; the other ten remain reachable in a one-lane world, because
they are about *when* an operation runs, not about *how many values* a reference
is made of.
