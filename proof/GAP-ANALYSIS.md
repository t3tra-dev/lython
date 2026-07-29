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

### Inexpressible — 11 when this was written; **6 have since become expressible**

> **Update.** `Proof.Program.*` and `Proof.Concurrent.*` were built in response
> to this section. The six placement defects are now **statable** — there is a
> step relation, block arguments, an unwind edge and reachability — and one of
> them is **exhibited**: `Proof.Program.Trace` computes
> `ownedCount env o ≡ suc (logicalRC (sites m) o)` after a branch, which is the
> SIGSEGV's root as an equation. Statable is not proved-absent; the rows below
> are kept as written, with the current status in the last column.

| defect (all measured on this tree) | what was missing | now |
|---|---|---|
| **`for` in `try` SIGSEGV** — `rc=139`, deterministic, on `try: for v in [1,2]: total += v except: total += 100`. Root: `for` allocates the cell once and threads it as a block argument, so **one allocation has two SSA names and the ownership machinery makes that two entities**; the unwind pad frees the cell before the handler in the same function reads it | CFG, unwind edges, **SSA identity** | **exhibited** — `one-object-two-names`, `names-and-sites-disagree` |
| **normal-path double dealloc** — the shipped compiler runs `__ly_dealloc___ly_cell_1` twice for the same cell; benign only because the freed read does not come back as 1 | **statable** |
| **loop `else`-only write lost** — `in else 7 / after 0`, silent, rc=0 | CFG | **statable** |
| **family C**: release placement abandoned when one consuming use existed | CFG + placement | **statable** |
| **families A/B**: latent double release on unwind edges no input reaches | CFG + unwind edges | **statable** — `step-invoke-throw` is the edge |
| **family D**: retain omitted by provenance, not layout | SSA provenance | **partly** — names exist; the IR does not yet record which op produced one |
| **family E**: block-argument index space vs successor-operand index space | CFG, block arguments | **statable** — `bindParams` refuses a length mismatch |
| **sequence/dict literal source move** — a use-**set** fact standing in for an execution-**frequency** fact | execution frequency | **partly** — loops are now expressible as a back edge; "how often" still is not |
| **read-back token** — a second owned token on the same SSA values | SSA aliasing + aggregates | **partly** — `Aliases` is exactly this; aggregates are still absent |
| **holder discharge / remaining leak families** — need a token **count**, not a token **name** | aggregates with multiplicity | **no change** |
| **deallocator selection**: 5 of 14 widths shared, `dict` in a 7-way tie | contracts | **no change** |

### Found after this document was written — 2, both closed

The two root causes behind the "unspellable borrow-edge retain" the compiler
carried. Both were diagnosed against the model and both are now theorems, so
they are kept apart from the counts above rather than folded into them.

| defect | what was missing | now |
|---|---|---|
| **retain inside the initialisation window** — `Ly_IncRef observed non-positive refcount`, three golden cases, from a retain anchored at a `memref.alloc` result whose prefix words are stored afterwards | a state between "storage exists" and "object exists" | **excluded** — `new` is split into `alloc` and `init`; `no-dup-in-the-initialisation-window` and its `drop` twin say the relation has no such step, `dup-resumes-after-init` says the window closes, and `window-is-well-formed` says the window itself is legitimate. `Proof.Program.Run.hoisted-retain-has-no-step` is the IR the compiler used to emit, refuted |
| **ownership taken and not recorded** — `boxRuntimeObject` retained the payload without marking the box owned, so an attribute-driven pass read a frame-owned box as borrowed and tried to synthesise a retain for a MOVE | a distinction between what is TRUE and what the IR RECORDS | **proved** — `Proof.Program.Recorded`. `edgeRetain : Maybe Mode → …` is `isOwnedIncoming` with its actual input, and its type is the finding: a pass cannot consult the truth. `not-recording-breaks-faithfulness` is the obligation the boxing path failed; `no-drop-of-a-borrow` is the opposite mis-record, which is what a blanket `owned` attribute would have introduced |

`Proof.Program.Run` carries both as computed facts on a state the step relation
produced: `ledgerAsShipped` emits a retain, `ledgerRepaired` emits nothing, and
`attrsSayBorrow` is one ledger with two truths, faithful to one of them.

---

## 2. The five structural gaps

### 2.1 There is no program — **CLOSED**

The model has **operations** and no **program**. `Proof.Memory.Properties` and
`Proof.RC.Properties` prove things about one operation applied to one state.
There is no step relation, no basic block, no unwind edge, no trace — checked,
not assumed: nothing in `src/` matches `Step`, `—[`, `BasicBlock` or an event
relation.

The note asks for this in §5 and it was skipped in favour of getting the state
model right first. That was defensible; it is now the binding constraint.

> **Closed.** `Proof.Program.Step` has instruction steps, terminator steps with
> the current block's terminator as a premise, an `invoke` unwind edge, and
> `_—→*_` with transitivity. `reachable-preserves-heap` is the first theorem
> stated over reachability rather than over one operation.
>
> One modelling bug was caught while building it: the first terminator rules did
> not read the current block's terminator, so control could go to any label.
> A step relation that lets control go anywhere makes every reachability
> theorem vacuous — the opposite of what the layer is for.

**Cost, exactly.** Six of the eleven inexpressible defects are *placement*
defects — the operation is right and the position is wrong. A model with no
positions cannot have a wrong one. `WFRC` preservation is unprovable for the
same reason: "reachable" is undefined.

### 2.2 There is no SSA layer — **CLOSED, and it exhibited the defect**

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
contexts.

> **Closed.** `Proof.Program.Env` makes a name primitive and `Aliases es x y` --
> two names, one entity -- is the sentence that was missing.
>
> And it does more than restate the defect. `Proof.Program.Trace` computes both
> counts after a branch and finds them **different**:
>
> ```agda
> owned-names-after-branch  : ownedCount envAfterBr theObj ≡ 2
> ghost-count-after-branch  : logicalRC (sites machAfterNew) theObj ≡ 1
> names-and-sites-disagree  : ownedCount envAfterBr theObj
>                              ≡ suc (logicalRC (sites machAfterNew) theObj)
> ```
>
> `bindParams` binds a name and does **not** occupy an owner site, because a
> block argument is not a new reference. So a pass reading the name count as the
> number of references to release emits **one drop too many** -- the
> over-release -- and a pass occupying a site per block argument emits **one
> retain too few**.
>
> The fix is in neither count. `br` must either not create an owning name (it is
> a MOVE) or must occupy a site (it is a DUP), and **the IR as written leaves it
> ambiguous**. That ambiguity is where the compiler's bug lives, and it is now a
> property of a datatype rather than a description in prose.

### 2.3 A leak cannot be stated — **CLOSED, by a different route than proposed**

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

> **Closed, and the paragraph above proposed the wrong route.** The leak is not
> stated as a converse field of `WFRC` and `temp` is still unused. It is a
> SEPARATE invariant over the two counts — `NameSiteCoherent es m = ∀ o →
> ownedCount es o ≡ ghostRC m o` — because `WFRC` is about sites and a leak is
> about sites *without names*, which no field of `WFRC` mentions. And the
> construction window is handled by splitting `new` into `alloc` and `init`
> (§"Found after this document was written"), not by a transient site: an
> uninitialised object is absent from the object table, so it has no life to be
> live.
>
> `Proof.Program.Leak` proves coherence preserved by every rule and hence
> `no-reachable-state-leaks`; `Coherence.leak-is-unreachable` applies it to the
> exhibited leak. The mutation that matters: making `br` a DUP (occupy without
> vacate) breaks it, so **the block-argument-is-a-move decision is what leak
> freedom rests on.**
>
> What it does not say: it does not say this compiler is leak-free. A leak needs
> an incoherent start or a transition the relation lacks, and there are exactly
> two of those — **function entry** (no callee frame) and **scope exit** (nothing
> discards an environment). So a leak is a boundary and not a placement, which
> is the opposite of families A–E.
>
> **⛔ That prediction was measured and is WRONG (2026-07-30).** The largest
> leaking golden is neither boundary nor placement: `LyObject_FromSlot` returns a
> reference (refcount initialised to 1, `ly.ownership.owned_results = [0]`) and
> `runtime-lowering` retains it anyway, so the counter is one high and the
> release — which is present, same value, same block chain — cannot get it to
> zero. It is an **over-retain**, unbounded, one per boxed slot read.
>
> The reason the theorem missed it names the next gap. Model `dup` is ATOMIC: the
> counter bump, the owner site and the name arrive together, so a redundant `dup`
> preserves both `WFRC` and coherence — the model says an extra retain is SAFE.
> The compiler's extra retain is a BARE counter bump, no site and no name, and the
> model has no rule for one. `Proof.Program.Recorded` models it as a real `dup`,
> so it gets the arithmetic right (`the-unrecorded-retain-bumps-the-counter`) and
> the consequence wrong. Closing it means splitting `dup` the way `new` was split
> into `alloc`/`init`, for the same reason: one instruction doing two things with
> no way to speak about the gap between them.

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
| **no refcount operation exists between `alloc` and `init`** — the step relation has no derivation for one, so it is not an operation the compiler may choose to emit | enforced by `prefixIsInitializedAtDefinition` (ABI/EntityHeaderPrefix.h), which is a **convention about producers**: it accepts call results and block arguments and declines everything else, so a new producer costs a counted omission rather than a wrong retain. Correct today; nothing checks it stays correct |
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

   > **Closed, and both halves of this line turned out wrong.** Not a direction
   > of `WFRC` but a separate two-count invariant, and the construction window is
   > the `alloc`/`init` split rather than a transient site. See §2.3.

5. **Erased `WellShaped`.** Small, and it removes the one way the box invariant
   can be misapplied.

6. **A refinement to the emitted IR.** Last, and only meaningful once (2) exists
   — otherwise it spans two levels at once.

Concurrency (§5–§6 of the note: threads, permissions, race freedom) is
deliberately not on this list. It is a large piece of work and **not** on the
path to any defect this compiler has actually shipped, all of which are
single-threaded.

---

## 5. Concurrency, as built

Modelled ahead of need, because the alternative is worse: bolting a thread pool
onto a step relation that assumed one thread means revisiting every ownership
rule. The sequential relation is now the one-thread case of the concurrent one.

Present: threads with **private environments and a shared machine** (which is
the whole difficulty in one line), a **nondeterministic** `Scheduled` -- a
relation and not a function, because safety must hold for every schedule --
spawn/join, an event history, program-order and spawn/join happens-before edges,
and `Race` as a conflict that nothing orders **in either direction**.

`Conflict` carries all six conjuncts as real propositions, including a
**constructive** overlap witness, so the shared byte can be pointed at. The
first draft had two of them as `⊤` placeholders, which would have made a
"conflict" derivable for any two events.

Absent, and deliberately: the **permission algebra**. Without it there is no
theorem that a well-formed program is race-free. `RaceFree` is the statement
such a theorem would prove; nothing proves it, and **`Race` being definable is
not the same as any program being shown free of one.**

## 6. Honest summary

Of the seventeen memory-safety defects this session found and measured:

- **1** is excluded by the redesign outright
- **5** have theorems in the model
- **11** were inexpressible when this was written; **6 are now statable and 1 is
  exhibited as a computed mismatch**, after the program layer was built
- **4 remain inexpressible**: aggregates with multiplicity, deallocator
  selection, and the two "partly" rows above

The two barriers that were ranked first and second in §4 are closed, and so are
§4.3 (`Proof.RC.Aggregate`) and §4.4 (§2.3 above). The sentence that used to sit
here -- "what remains at the top of the list is aggregates with multiplicity" --
is stale and is kept only as a record of what was believed: §4.3 is done and the
top of the list is now §4.6, **refinement**, which cannot be closed inside
`proof/` at all.

The leak result changes the reading of the two open leak families rather than
closing them. They were characterised as needing a token *count* rather than a
token *name*; `no-reachable-state-leaks` says no instruction sequence produces a
leak, so the count is needed at a **boundary** -- function entry or scope exit --
and not along a path. That is a prediction, not a measurement.

Two things should not be read into the current state. The proof directory does
**not** yet constrain the compiler — no refinement exists, and the README says
so. And the one-lane object is a redesign whose value is demonstrated against
one defect class; the other ten remain reachable in a one-lane world, because
they are about *when* an operation runs, not about *how many values* a reference
is made of.
