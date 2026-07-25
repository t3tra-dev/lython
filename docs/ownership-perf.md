# Ownership verification: compile-time cost

The path-sensitive affine ownership verifier
(`verifier/runtime/AffineOwnership.cpp`, phase 13 of
`Passes/LoweringPipeline.cpp` and re-run at phases 10b and 14) proves the
affine invariant of `rfc/memory-safety-proof.md` over CFG exits. It is the
single most expensive verifier in the pipeline, and before the work recorded
here it was the reason `import string` cost ~17x the compile time of a program
with no import at all — and cost it superlinearly, so the multiplier grew with
the program (the Wave 2.5 hand-off measured ~80x on a larger case).

This file records what the cost actually was, what changed, and how the
changes were shown not to weaken the proof. It is not a change log; the
numbers are here so a future regression is visible as a regression.

## What was superlinear

Measured on `dda1cd8`, `lyc --emit-llvm`, Apple M-series, `RelWithDebInfo`.
Three independent superlinear terms, in the order they dominated:

1. **Visited-state membership was a linear rescan.** `verifyResourceOnCFGPaths`
   kept its visited path states in a flat vector and asked
   `containsPathState` — a full scan under `samePathState` — for every state
   popped off the worklist. One resource's walk was therefore quadratic in its
   own state count, up to ~2·10^8 state comparisons before the 20000-state cap
   fires. Same shape in `verifyBorrowedEntryOnCFGPaths`.

2. **Callee resolution was a module symbol-list walk.**
   `own::collectOwnedCallResultGroups` and `AliasAnalysis::build` each resolved
   a callee with `module.lookupSymbol<func::FuncOp>()` once per call op.
   The static `SymbolTable::lookupSymbolIn` scans the symbol-table op's block
   for a matching name — it builds no index — so both sweeps were
   **O(calls × symbols)**, and importing a stdlib module raises *both* factors
   at once. This is the term that made an import cost so much more than the
   same code written inline.

3. **Each resource walked the whole reachable CFG, released or not.** A
   resource released two ops after its producer still walked every remaining
   op of the function — once per resource. With one owned resource per
   object-producing call, resources and ops both grow with function size, so
   this term was **O(resources × ops)**, i.e. quadratic in `__main__`.

Below those, per visited path state the walk re-asked questions whose answers
are fixed by the IR: five `FuncContractCache` lookups per call op (a string
hash each), `precedingTryCallSiteMarker` / `guardedCallAfterMarker` /
`anchorTrueEdgeGuardedCall` / `mayRaisePythonException` recomputed from
scratch, a full `region.walk` per region-carrying op, and an
`O(|group| × |retain operands in function|)` scan at every may-raise call.

## What changed

All of it in `verifier/runtime/AffineOwnership.cpp`, plus the callee-resolution
fix in `common/Ownership.cpp`. No judgment moved: see "Soundness" below.

- `VisitedAffineStates` / `VisitedBorrowedStates` — hash-bucketed membership
  over the *same* `samePathState` / `sameBorrowedPathState` relations. The hash
  covers exactly the fields those relations compare, so states that hash apart
  are provably distinct under them; the cap and its diagnostic are unchanged.
  The states themselves stay in one flat insertion-ordered vector and the
  buckets hold indices into it — a path state carries four small value vectors,
  and putting whole states inside hash buckets paid the map's capacity slack on
  all of it (that alone was +170 MB of peak RSS).
- One `mlir::SymbolTable` per verifier invocation, threaded into
  `collectTrackedResources`; `AliasAnalysis::build` builds one locally.
  `collectOwnedCallResultGroups` gained an optional trailing
  `mlir::SymbolTable *` so the insertion side can adopt it without a signature
  break.
- `OwnershipWalkCache` — a per-function memo for facts that are pure functions
  of IR the verifier only reads: an operand→alias-class inverse
  (`classUsers`), the marker/anchor/raise-classification answers, and the
  region-mentions-group query. It lives in the verifier, not beside the `own::`
  helpers, because the refcount-insertion pass calls those helpers while it
  rewrites marker ids and adds blocks.
- The per-op predicate battery is guarded by `mentionsTracked`: every
  group/stale/previous/views predicate requires an operand aliasing a tracked
  value, so an op that mentions none of them answers `false` without being
  asked. The unwind-exit checks are *not* guarded — they are properties of the
  callee, not of the operands.
- **Released-path pruning.** A path whose token is `Released` owes nothing at
  any exit, and every diagnostic still reachable on it (double release, use
  after release, partial consume, release through a transferred name) needs an
  op naming a tracked value. When mention-reachability proves no such op is
  reachable, the path is dropped. This is what removes term 3.

## Numbers

`lyc --emit-llvm <case> -o /dev/null`, `LYTHON_PERF=1` phase totals in seconds,
peak RSS in MB. `lowering` is the whole pipeline; `path-sensitive` is
`func-call-ownership.path-sensitive` summed over its invocations. Before is
`dda1cd8`. The two binaries were run alternately on the same inputs so that
machine load drifts hit both.

| case | lowering | path-sensitive | peak RSS |
| --- | --- | --- | --- |
| `examples/hello.py` | 0.81 → 0.75 | 0.01 → 0.01 | 92 → 92 |
| `cases/stdlib_string_basics.py` | 4.85 → 3.54 | 1.30 → 0.51 | 195 → 180 |
| `cases/stdlib_string_catch.py` | 8.98 → 6.83 | 2.21 → 0.52 | 218 → 217 |
| `cases/stdlib_string_template.py` | 13.97 → 8.55 | 5.42 → 0.76 | 317 → 317 |

Scaling in user code size: `from string import Template` plus N
`Template(...)` / `safe_substitute(...)` statement pairs.

| N | lowering | path-sensitive | peak RSS |
| --- | --- | --- | --- |
| 1 | 3.15 → 2.40 | 0.78 → 0.41 | 166 → 168 |
| 2 | 3.99 → 3.05 | 1.05 → 0.49 | 173 → 179 |
| 4 | 6.83 → 4.86 | 2.15 → 0.66 | 236 → 202 |
| 8 | 22.56 → 13.49 | 8.00 → 0.98 | 381 → 320 |
| 16 | 122.78 → 108.73 | 46.68 → 1.87 | 724 → 714 |

The verifier's growth over that range drops from roughly cubic to roughly
linear. Per doubling of N the before column multiplies by 1.35, 2.05, 3.72,
5.84; the after column by 1.20, 1.35, 1.48, 1.91. At N=16 the phase is 25x
faster, and peak RSS is at parity — the flat visited vector paid for the
caches.

The `lowering` column improves much less than the verifier does because
`lowering.post-cleanup-unwind-insertion` now dominates it (see "Still open").
That phase measured 64.43 before and 94.88 after at N=16, but it swings by well
over 30% run to run on a loaded machine (repeat runs of the same binary gave
70.1 and 107.8), so treat its column as noise rather than as an effect of this
change: nothing here touches the insertion pass except the shared
`AliasAnalysis::build`, which can only make it faster.

## Soundness

The proof kernel is unchanged: no judgment of `rfc/memory-safety-proof.md` was
relaxed, no check was made conditional on anything but a proof that its answer
is already determined. The three arguments that carry the work:

1. **Hashing the visited set** preserves the membership relation exactly. The
   hash is a function of precisely the fields `samePathState` compares, so
   equal states land in the same bucket and the bucket is compared with the
   unchanged predicate. Worklist and insertion order are untouched, so the
   *first* diagnostic emitted for a module is the same one.
2. **`mentionsTracked` guards** are short-circuits of predicates that all
   require an operand aliasing a tracked value. An empty group deliberately
   reports as mentioning, because `groupMatchesValues` matches an empty group
   vacuously and filtering on it would answer differently from the predicate.
3. **Released-path pruning** rests on two facts. (a) With the token released,
   every exit rule succeeds and every remaining `emitError` on the path is
   gated on a mention of a tracked value — this was checked site by site.
   (b) The mention-reachability query over-approximates in every direction:
   block granularity (a mention anywhere in a block counts, including before
   the walk's position), block-level exception edges instead of per-marker
   ones, and nested-region ops attributed to their enclosing top-level block.
   Its edge set is a superset of the walk's own transitions, so `false` is a
   proof, not a guess. Renaming is covered by the same argument: the walk can
   only rename a group at a terminator that forwards it, and such a terminator
   *is* a mentioning op, so a state that would later rename is never pruned.

One behavioural difference is accepted and recorded: the
`ownership CFG exploration exceeded 20000 states` bail-out fires on strictly
fewer inputs, because fewer states are explored. That message is a resource
bound, not a proof obligation — it reports that the verifier gave up — so
firing it less often removes spurious hard errors and never admits a leak.

How it was checked:

- `ctest` 381/381, including the two `errors/` cases that pin an ownership
  diagnostic — `try_dict_merge_rebind` and `try_structural_rebind` both expect
  `released owned resource .* is used after release`, which is exactly the
  diagnostic class the released-path prune could have hidden.
- Exact `stderr` and exit-code diff, before vs after, over all 328
  `tests/golden/cases/*.py`, `tests/golden/errors/*.py` and `examples/*.py`:
  no divergence. Stronger than the golden runner, which matches `stderr`
  against a regex — here the two ownership diagnostics above are byte-identical
  down to the source location.
- A/B differential with the released-path prune disabled (a temporary switch,
  not shipped) over the same 328 inputs: identical exit codes and `stderr`.
  No input in the corpus has a diagnostic that only the unpruned walk finds.
- Emitted LLVM IR compared before vs after over the 240 corpus inputs that
  compile: byte-identical. The verifiers do not rewrite IR, and the shared
  `AliasAnalysis::build` change only swaps how a callee symbol is resolved, so
  identical IR is the expected result and a divergence would have meant the
  symbol-table lookup disagreed with `lookupSymbol`.
- ASan fuzz regression (`ctest --test-dir build-fuzz`) 3/3: the checked-in
  corpora plus the golden cases, replayed unmutated.

## Still open

`lowering.post-cleanup-unwind-insertion` is now the dominant phase — most of
the `lowering` total at N=16 above. The cause is
`UnwindCleanupAnalysis::reachesAvoiding`
(`Passes/Runtime/Passes/Ownership.cpp`): a fresh BFS with a fresh visited set
per call, asked once per (consume site × point). The query is separable — `to`
is only a membership test inside the BFS loop, never a cutoff — so caching
`reachableAvoiding(from, avoid, fromAfter)` collapses the inner loop to one
BFS per consume site. That is insertion-pass work and is tracked with the
unwind landing-pad track, not here.

Secondary, in rough order: the remaining `getInherentAttr` traffic from
`readFunctionContract` being re-read rather than cached per callee;
`AliasAnalysis::track` invalidating the alias buckets on every newly tracked
value; and the thread-safety verifier's own per-call
`module.lookupSymbol<func::FuncOp>()` in `ThreadSafe.cpp` /
`ThreadSafeModel.cpp`, which is the same O(calls × symbols) shape on a much
narrower path (`lowering.thread-safety-verifier`, ~0.8s of the 8.55s
`stdlib_string_template.py` total).
