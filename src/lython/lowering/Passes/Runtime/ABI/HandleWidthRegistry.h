#pragma once

// The assignment of one-lane handle widths, and a ledger of where that
// assignment has FAILED. A stage-4b conversion (rfc/object-ownership-kernel.md)
// records its `memref<Nxi64>` here in the same commit that flips
// `ly.runtime.shape`, and reads the file first rather than grepping one tree for
// absence -- a width can already be taken by a track that has not merged, which
// is how `builtins.complex` collided with `builtins.bytes` twice, at 4 and then
// at 6 after that track moved off 4.
//
// ===========================================================================
// READ THIS FIRST: width uniqueness is an INTERIM LOCAL MITIGATION, not the
// protection scheme. The durable fix is contract-name availability.
// ===========================================================================
//
// Uniqueness does not scale, and the arithmetic is short enough to check. There
// are **24 single-input deallocators** today against roughly **six** free widths
// (9, 11, 13, 14, 15, and 6 held by `bytes` in flight). Still to convert:
// `list`, `set`, `frozenset`, `tuple`, `int`, `str`, ~70 exception contracts, and
// class instances. Unique widths cannot be allocated for those. And a width IS
// object size -- `complex` pays 32 -> 56 B for identity alone, three dead words
// on every allocation.
//
// ===========================================================================
// AND IT HAS NOW RUN OUT. That paragraph was written when the arithmetic was a
// projection; `list` took 9, `set` 11, `frozenset` 13, `tuple` 14 and `int` is
// reserved at 15, so the free list is **EMPTY** and `builtins.str`, the ~70
// exception contracts and class instances have no unique width available. The
// full statement, including why "pick a bigger number" is not a way out, is in
// the boxed note under the assignment table -- it is not repeated here, but the
// count above ("roughly six free widths") should be read as historical.
//
// It ran out in four container conversions, exactly as the 24-vs-6 arithmetic
// said it would, which is the one thing this file was built to make visible
// before rather than after the fact.
//
// ⛔ AND THE REMAINING WORK IT IS COUNTING IS NOT WHAT IT SAYS. "`str`, ~70
// exception contracts, and class instances" is, measured (see the block after
// int's, at the end of this file): ONE deferred conversion (`str`), ONE
// declaration rather than ~70 (the whole exception family shares
// `LyBaseException_DecRef`; 66 of the 67 contracts declare only a shape, and
// those 66 shapes are dropped by `collectRuntimeDeallocators` for want of a
// deallocator to join to), and ONE non-item (class instances never choose a
// width -- their handle is `memref<16xi64>` by construction). Both real
// conversions were measured to reduce ambiguity by zero. So the free list being
// empty is not the thing blocking the work; the work is not there.
// ===========================================================================
//
// ===========================================================================
// ⛔ AND THE "FREE LIST" WAS THE WRONG BOOKKEEPING. Measured 2026-07-28 by
// tests/probe/tools/tiecensus.py, which reads the manifests instead of this
// file: **5 of the 14 single-input deallocator widths are claimed by more than
// one contract, and four contracts recorded above as converted-and-safe are in
// a tie right now.**
//
//     width 8   7-way   _io.BytesIO, _io.FileIO, _io.StringIO,
//                       _io.TextIOWrapper, asyncio.AbstractEventLoop,
//                       **builtins.dict**, builtins.function
//     width 5   3-way   **builtins.range**, **builtins.range_iterator**,
//                       types.CoroutineType
//     width 3   3-way   builtins.bool, **builtins.float**,
//                       lyrt.ReadyIntAwaitable
//     width 2   3-way   builtins.int, builtins.str, contextlib.nullcontext
//     width 4   3-way   lyrt.AsyncCounter, lyrt.Counter,
//                       lyrt.ReadyAsyncCounter
//
//     unique (9): list 9, set 11, frozenset 13, tuple 14, bytes 6, complex 7,
//                 object 16, _asyncio.Future 10, types.GeneratorType 64
//
// This file tracked the widths I had ASSIGNED and I read that as the widths
// that were UNIQUE. `dict`, `float`, `range` and `range_iterator` were never
// unique; they were recorded as converted and the collision was never checked.
//
// Why that matters more than the bookkeeping error: **sharing a width is the
// common case, not the exceptional one** -- 16 of 28 single-input deallocators
// sit in a tie. So a tie is NECESSARY but NOT SUFFICIENT for the defect the
// set/frozenset conversion fixed; that one also needed a transfer contract
// feeding a stale lane. The four remaining ties therefore need auditing for the
// sufficient part, and "it is in a tie, convert it" is not the conclusion.
//
// The census is a script and not a comment on purpose: this block was wrong for
// as long as it was maintained by hand, and it is exactly the kind of claim
// that reads as verified because it is written down.
// ===========================================================================
//
// So: uniqueness rescues one contract when a width happens to be free. It cannot
// be the scheme. **The scheme is that the contract-aware overload is always
// taken, so neither `common/Ownership.cpp:429` nor `:450` -- the two paths that
// fall back to structural matching -- is ever reached.** Two closable gaps stand
// between here and that, both measured:
//
//   GAP 1: ~40% of owned results name no result contract at all.
//     Of 680 `ly.ownership.owned_results` declarations: 372 carry
//     `ly.runtime.result_contract`, 22 `ly.ownership.owned_result_contracts`,
//     17 `element_`/`next_contract`. (374 for the first on this branch, which
//     adds two.) Every declaration without one is a site where the name is
//     simply unavailable and the structural fallback is the only option.
//
//   GAP 2: some readers consult only `owned_result_contracts` and lack the
//     `ly.runtime.result_contract` fallback, so the SAME declaration yields two
//     different answers depending on who asks. The insertion path has both
//     (`common/Ownership.cpp:1073-1082`, with the reason in a comment). These
//     three sites have only the first:
//         verifier/runtime/AffineOwnership.cpp:148  (returnCarriesGroupInsideOwnedAggregate)
//         common/Ownership.cpp:995                  (staticEvidenceCoveredLogicalOffsets)
//         common/Ownership.cpp:1311                 (unionStaticEvidenceCallResultAliases)
//     Two of these three labels were wrong until 2026-07-28 -- :995 was named
//     after callableOwnedReturnRanges and :1311 after the function that is
//     actually at :995, and the verifier site was cited at :61, which is
//     ownershipStaleTraceEnabled. Verified against the code: the reads are at
//     :995/:999, :1311/:1315 and AffineOwnership.cpp:149-150. A comment naming a
//     line number decays the moment anything above it moves, which is why the
//     function name is the part to trust and the line the part to re-check.
//     Scope correction worth keeping: this is NOT "the verifier gets a name for
//     22 of 680". The verifier's main resource discovery goes through the shared
//     `collectOwnedCallResultGroups`, which HAS the fallback; it is these three
//     helpers that do not. The direction of the resulting error is untraced --
//     for the first, a null deallocator makes the predicate answer "no", which
//     looks like it would produce a spurious refusal rather than a silent
//     acceptance, but that has not been followed to a caller.
//
// Closing GAP 1 and GAP 2 makes every width tie below harmless. Until then the
// ties are live, and the ledger of them is the point of this file.
//
// WHY ONE-LANING IS EXPOSED IN THE FIRST PLACE, and why the fix is the name
// rather than the width.
//
// `shapeMatch` scores 0 unless `shapeTypes.size() > inputTypes.size()`. A
// multi-lane contract satisfies that by construction: its canonical shape is
// several types and its release interface is an entity-root PREFIX of them,
// hence shorter. `builtins.str` is shape 2 / interface 1, so it scores 2 -- and
// that positive score is what lets it WIN the tiebreak against a same-width
// group of zero-scorers. A one-lane contract is shape 1 / interface 1 and scores
// 0 always; it cannot win a tiebreak, because the possibility is structurally
// absent.
//
//     Converting a contract to one lane therefore strips it of its shapeMatch
//     protection, and the ONLY protection left is the contract NAME.
//
// That is the mechanism behind the 28-vs-0 measurement below -- not a
// coincidence of that program but a structural consequence of arity -- and it is
// also why GAP 1 and GAP 2 above are the fix. A unique width substitutes for the
// lost protection *for one contract*; making the name always available restores
// it for all of them. Uniqueness is the local rescue, the name is the scheme.
//
// READ, NOT MEASURED -- and it implicates already-shipped code. All seven
// contracts at width 8 have shape arity == interface arity, so all score 0:
//
//     dict, _io.{StringIO,BytesIO,TextIOWrapper,FileIO},
//     asyncio.AbstractEventLoop   shape memref<8xi64>, decref (memref<8xi64>)
//     builtins.function           no shape,            decref (memref<8xi64>)
//
// A bare `memref<8xi64>` therefore matches all seven, all at 0, all with
// `inputTypes.size() == 1` -> `ambiguous` -> `nullptr`. **`builtins.dict`, the
// first contract converted and already on main, has none of the protection
// `str` still has**, and rests entirely on the name being available -- which
// GAP 1 says it is not, at ~40% of owned results.
//
// MEASURED SINCE (playbook 7.14): that reasoning is sound about the algebra and
// unconfirmed about the traffic. A dict program produced ZERO ambiguous exits at
// `memref<8xi64>` and zero structural resolutions of `builtins.dict`, so the
// contract-named path is in fact carrying dict. The gap between "can be reached"
// and "is reached" is still open -- but it is open in the reassuring direction,
// and no program has yet been found that reaches it.
//
// The same now applies to `builtins.float` (width 3) and
// `builtins.range`/`range_iterator` (width 5) from this branch: see the
// known-live ledger below. They are in the same position as `dict`, not a
// better one.
//
// This is derived from the algebra and the type arities. Whether any of these
// contracts actually REACHES the contract-less path is NOT measured. That is the
// highest-priority item outstanding on this file, it now has to answer for
// `float` and `range` as well as `dict`, and note what it adjudicates: shipped
// code, not a future change. The instrument is the one that produced 28-vs-0 --
// count the contract-less `return nullptr` and the structural resolutions for
// the contract over a few lines of code that use it, then compare against the
// contract moved to a free width.
//
// Why a width is not a style choice. Sharing one has TWO distinct failure modes,
// and they present differently -- a converter debugging one will look for the
// wrong symptom if the two are conflated. Both come out of the selection loop in
// `findDeallocatorForValueGroup` (`common/Ownership.cpp:386`):
//
//   (A) AMBIGUITY -> no group forms.
//       Equal `inputTypes.size()` and equal shape score sets `ambiguous = true`,
//       and the function returns nullptr. Symptom: the resource is never
//       discovered, so no release is inserted at all. Presents as a leak, or as
//       an ownership obligation that nothing ever refuses.
//       Applies to a width taken as a SINGLE-INPUT release interface.
//       MEASURED, by the str/bytes track, instrumenting that `return nullptr`
//       over the three-line program `b = b"hello"; len(b); b + b"!"`:
//
//           bytes at memref<4xi64>, tied with 3 lyrt.* .. 28 nullptr,  0 resolved
//           bytes at an untied width ..................... 0 nullptr, 21 resolved
//
//       This is how the `str`/`bytes` tie hid the exception family's
//       `owned_results = [0, 1]` double free: never instantiated, so nothing
//       could refuse it, one shape away from shipping as a use-after-free under
//       `--release`.
//
//   (B) PREEMPTION -> the WRONG deallocator is selected.
//       A longer `inputTypes.size()` wins outright and RESETS `ambiguous` to
//       false, so no tie is ever recorded. A width appearing as a component of a
//       multi-input release interface therefore does not tie with a one-lane
//       contract; if the longer interface matches at that offset it takes the
//       match. Symptom: a release through another contract's deallocator, with
//       no diagnostic.
//       Applies to a width appearing INSIDE a multi-input release interface.
//
//       ✅ NOW MEASURED (2026-07-28) AND EMPTY: 0 genuine instances.
//       tests/probe/tools/preemption.py replays the selection loop over the
//       manifests, because the C++ census structurally cannot count this -- the
//       longer interface winning RESETS `ambiguous`, so a preemption leaves no
//       trace to count. Over 1103 functions with results: 25 preemptions whose
//       winner the callee declares (benign), 206 where the callee's contract
//       declares no deallocator so the winner is inherited by design (exception
//       subclasses under LyBaseException_DecRef), and 3 residual where the
//       callee owns no declared result contract but the winner is manifestly
//       right -- LyFutureIter_New -> _asyncio.FutureIter, LyTaskIter_New ->
//       _asyncio.TaskIter, LyStopAsyncIteration_New -> BaseException. In all
//       three the losers are bool/float/lyrt.ReadyIntAwaitable, i.e. the width-3
//       tie IS on the preemption path and the longer interface that wins is the
//       correct owner.
//
//       The instrument's first answer was 201 preemptions at width 3, and that
//       was a false positive of the checker: for a genuine exception triple the
//       leading memref<3xi64> IS the exception header, so "a shorter interface
//       also matched" is true of every correct case. That is the same
//       width-is-not-proof-of-kind fallacy this file documents, one level up, in
//       the thing checking for it. Adjudicating by declaration rather than by
//       shape is what took it to 0.
//
// Do not confuse (B) with a third mechanism that shares its symptom:
//
//   (C) A WRONG CONTRACT NAME, made survivable by a width coincidence.
//       The contract-NAMED overload (`common/Ownership.cpp:424`) filters
//       candidates by exact `contractName`, so if the caller supplies a name
//       that is not the resource's, and the named contract's release interface
//       happens to type-check against the value, there is exactly ONE candidate:
//       neither the length comparison nor the ambiguity flag is reached, and the
//       wrong deallocator is returned. That is the `LyFloat_Repr` defect --
//       `collectContractOwnedResultGroups` named an owned `builtins.str` result
//       by the RECEIVER's contract, and `LyFloat_DecRef`'s `(memref<2xi64>)`
//       matched a str header. **It is not preemption**: no second candidate
//       existed. It is fixed by declaring `ly.runtime.result_contract`, not by
//       choosing a width -- though a width coincidence is what let it go
//       unnoticed, because every contract's interface was `(memref<2xi64>)`.
//       So when a release goes through the wrong contract's deallocator, check
//       the declared name BEFORE suspecting the width.
//
// The earlier argument for tolerating a tie -- that the shape tiebreak is
// guarded by `shapeTypes.size() > inputTypes.size()`, false for every one-lane
// contract, so no one-lane contract can win it anyway -- is WITHDRAWN. It
// correctly predicted that a tie among one-lane contracts cannot mis-attribute,
// and missed (A) entirely: a tie does not need to mis-attribute to do damage.
//
// Scope: a collision is on the RELEASE INTERFACE, not on the canonical shape.
// For a one-lane contract the two coincide, so the width below is both.
//
// This header is documentation, like the word assignments beside it: nothing
// includes it, because no C++ addresses these words -- a fixed-width inline
// payload is reached only through the contract's own accessors. Its value is
// that a collision becomes a review-time conflict in ONE file instead of a
// merged-tree discovery that neither branch's author can make alone.

namespace py::lowering::handle_width {

// ---------------------------------------------------------------------------
// Assigned. Keep sorted by width.
//
//   width  contract(s)                                      state
//   -----  -----------------------------------------------  -------------------
//       2  builtins.int, builtins.str,                      NOT AVAILABLE:
//          builtins.bytes(pre-conversion),                  multi-lane contracts
//          contextlib.nullcontext                           share this release
//                                                           interface 7 ways
//       3  builtins.float                                   one-lane (converted)
//          builtins.bool (boxed), lyrt.ReadyIntAwaitable    pre-existing tie
//       4  lyrt.Counter, lyrt.AsyncCounter,                 NOT AVAILABLE:
//          lyrt.ReadyAsyncCounter                           3-way tie
//       5  builtins.range, builtins.range_iterator          one-lane (converted)
//          types.CoroutineType                              pre-existing tie
//       6  builtins.bytes                                   one-lane, MERGED
//                                                           in 00079ef; sole
//                                                           owner
//       7  builtins.complex                                 one-lane (converted)
//       8  builtins.dict, builtins.function,                7-way tie, all
//          _io.{StringIO,BytesIO,FileIO,TextIOWrapper},     one-lane already
//          asyncio.AbstractEventLoop
//       9  builtins.list                                    one-lane, sole
//                                                           owner. Layout needs
//                                                           5 words; 9 is the
//                                                           narrowest FREE
//                                                           single-input
//                                                           interface, and it
//                                                           must be >= 8 for
//                                                           isContainerHandleType
//      10  _asyncio.Future                                  one-lane
//      11  builtins.set                                     one-lane, sole
//                                                           owner, CONVERTED.
//                                                           Layout needs 5 words
//                                                           (words 0..7 of
//                                                           ContainerLayout.h,
//                                                           5 and 6 unused by a
//                                                           set); 8..10 are dead
//                                                           space bought for an
//                                                           untied interface
//      12  _asyncio.Task                                    leading word of a
//                                                           multi-input release
//      13  builtins.frozenset                               one-lane, sole
//                                                           owner, CONVERTED.
//                                                           Same layout as
//                                                           `set`; a DIFFERENT
//                                                           width on purpose --
//                                                           see below
//      14  builtins.tuple                                   one-lane, sole
//                                                           owner. Layout needs
//                                                           5 words (words 0..7
//                                                           are ContainerLayout,
//                                                           5/6 unused by a
//                                                           sequence); words
//                                                           8..13 are dead space
//                                                           bought for an untied
//                                                           single-input release
//                                                           interface
//      15  builtins.int                                     RESERVED, NOT YET
//                                                           CONVERTED, AND
//                                                           DEFERRED ON PURPOSE
//                                                           -- see the note at
//                                                           the end of this
//                                                           file. The
//                                                           reservation is now
//                                                           machine-checked by
//                                                           ctest
//                                                           `abi.handle_width_reservations`
//      16  builtins.object                                  payload box / boxed
//                                                           field slot
//      64  types.GeneratorType                              frame
//
// WHY `set` AND `frozenset` DO NOT SHARE A WIDTH, when they share a layout
// exactly. Sharing one would leave them tied with each other on release
// interface, which is precisely the `range`/`range_iterator` situation this
// file calls "the clearest demonstration that uniqueness is the wrong axis".
// The difference is that `range`/`range_iterator` are DELIBERATELY structurally
// identical -- nothing distinguishes them, so no assignment can -- whereas
// `set` and `frozenset` are distinguishable (mutability, `__hash__`, class id
// 21 vs 23). A tie that a width CAN break is worth breaking while a width is
// still available; a tie that no width can break is the argument against the
// scheme. Two widths were spent here rather than one for that reason, and it is
// half of why the free list below is empty.
//
// The cost of the separation is structural rather than in words: the frozenset
// wrappers used to hand their own three lanes straight into the `LySet_*`
// bodies, which only worked because the two canonical shapes were
// byte-identical. They cannot now, so every algorithm lives once in a helper
// taking the handle as `memref<?xi64>` (a `memref.cast` of either width) and
// the per-contract functions are thin wrappers over it. That is the same
// "parameterise the one loop" move `68feee7` made for the shared sequence
// helpers, for the same reason: two copies of a probe loop drift apart while
// both keep compiling.
//
// ===========================================================================
// THE FREE LIST IS NOW EMPTY. This is the arithmetic at the top of the file
// arriving, and it is worth stating as a fact rather than a projection.
//
// Widths 11, 13, 14 and 15 were the last four free single-input release
// interfaces. They are spoken for: `set`->11, `frozenset`->13, `tuple`->14,
// `int`->15. **Nothing is left for `builtins.str`, the ~70 exception contracts,
// or class instances** -- the three largest items remaining in stage 4b, and
// the one (`str`) whose conversion §8.3 adjudicated as DEPENDING on a unique
// width (one-laning it drops its `shapeMatch` from 2 to 0 and sweeps 27 sites
// into the eight-way `memref<2xi64>` ambiguity -- READ off the algebra, not
// measured, and §8.3 says so).
//
// ⛔ Correction (2026-07-28, measured): `str` is NOT the only contract at width
// 2 with a positive `shapeMatch`. `int`'s shape is the 3-lane
// `(memref<2xi64>, memref<2xi64>, memref<?xi32>)`, so it scores **3** against a
// 1-input interface -- protected more strongly than `str`'s 2. So converting
// `int` destroys a larger score than converting `str` does, and both
// conversions, not just `str`'s, drop into width 2's ambiguity. Whichever is
// converted first, the other's protection is unaffected; what changes is only
// its own.
//
// ✅ And the set-shaped SUFFICIENT condition is absent from every remaining tie
// (measured 2026-07-28). What the set/frozenset defect needed on top of a shared
// width was a **mutable interior buffer held as a separate SSA lane**, so a
// holder could keep a lane the reallocation freed. Auditing shapes: widths 3, 4,
// 5 and 8 members are all single-lane (`shapeTypes == inputTypes`), so no
// interior address exists as a lane to go stale; width 2's `int` and `str` do
// have interior lanes but are immutable, so nothing reallocates. **No member of
// any remaining tie has both properties.** A tie is therefore necessary and not
// sufficient, and this is the structural reason rather than an absence of
// symptoms -- which matters, because the set defect exited 0 and was silent.
//
// So the interim mitigation has run out before the work has. Those three must
// either land on a tied width and accept mechanism (A), or GAP 1 and GAP 2 must
// be closed first. **The scheme was always the contract name; this is the point
// at which there is no longer a local rescue to fall back on.** Do not read the
// empty free list as "pick a bigger number": width is object size, the ties are
// on the release interface rather than the shape, and a contract at width 20
// still scores 0 and still cannot win a tiebreak -- it just delays the
// collision while paying for it on every allocation.
//
// The `set`/`frozenset` and `tuple` tracks wrote this paragraph independently,
// from opposite ends of the same four-way tie, and agreed. That is worth one
// line of its own: the exhaustion is not one track's editorial framing.
// ===========================================================================
//
// ---------------------------------------------------------------------------
// COLLISIONS, and their measured reachability. Occupancy of single-input
// deallocator interfaces on the merged tree.
//
// IMPORTANT, and a correction to an earlier version of this file: these ties are
// NOT known-live suppressions. Instrumenting the contract-less overload and
// attributing each ambiguous exit BY TYPE (playbook 7.14) found **zero**
// ambiguity on any one-lane width -- 3, 5, 6, 7 or 8 -- across four programs,
// one per contract. One-lane contracts resolved structurally ZERO times, which
// is the positive evidence for the shapeMatch mechanism above: they are never
// put in a structural tiebreak because the contract-named path answers first.
//
// ** CORRECTION to the sentence that used to follow (measured on main 5907b97
// by the list track, keying each ambiguous exit on ALL of the tying value
// types rather than on the leading one). "All 119 ambiguous exits were on
// `memref<2xi64>`" is wrong, and it is wrong in a way that hid the ONE tie a
// conversion was about to break: **
//
//     ambiguous on (memref<2xi64>)                              =  56
//     ambiguous on (memref<2xi64>, 2xi64, ?xi64)                =  63
//                                                                 ---
//                                                                 119
//
// The three-type group is the four-way list/tuple/set/frozenset tie, whose
// FIRST type is `memref<2xi64>` -- so an instrument that printed only
// `values[offset]` filed all 63 under width 2. The total is unchanged, which is
// why the two readings look like the same measurement. Same trap as the "width
// is not a proof of kind" fallacy below, one level up: an ARITY-1 key cannot
// name an arity-3 tie.
//
// Both figures are invariant across ten programs (six single-contract probes
// plus four list/set-heavy goldens), i.e. they are constants of lowering the
// runtime library rather than products of the program -- `nomatch` and
// `resolved` do move, which is what shows the counter reads the input at all.
// One program moved `ambiguous`: `golden/cases/list_methods.py` reports **137**,
// and the extra 18 are on `(memref<3xi64>)` -- the width-3 tie of `LyBool` /
// `LyFloat` / `lyrt.ReadyIntAwaitable`, reached by a list holding both floats
// and bools. **So width 3 is not "a tie four programs failed to reach": a
// program reaches it.** That is an item for whoever owns `float`, not for this
// file to fix.
//
// Also measured and worth having: `contractName.empty()` at
// `common/Ownership.cpp:428` fires **0** times in all ten programs, and the
// `:450` fallback fires 1078-1312 times per compile but produces **no**
// ambiguous exit -- every one of the 119 comes from a call site that enters the
// contract-less overload DIRECTLY. So GAP 1 does not present as an empty
// contract name at this call site; it presents as call sites that never had one.
//
// ⛔ The `:450` figure is STALE as of `bcfbbf9` (measured 2026-07-28 with the
// permanent `LYTHON_DEALLOC_CENSUS=1` instrument): `fallback_450` is **0**, not
// 1078-1312, and so are `empty_name` and `contract_aware_ambiguous`. Site A
// preferring the declared name (`5c52ae0`) is what closed it -- the named
// overload now type-matches on the first try, so nothing falls through. The
// surviving GAP-1 surface is `declared_name_absent = 308`, invariant across every
// program measured. Quote 308, not 1078-1312, and note that a `:450` count of 0
// means the fallback is no longer the path to instrument.
//
// ⛔ AND "THE WIDTH-2 AMBIGUITY IS N" IS NOT A PROPERTY OF THE TREE. This file
// records 21 and `rfc/memory-safety-proof.md` records 42, each as though it were
// the figure. Measured over **294 programs** (291 golden cases plus three
// probes), exit code checked on every one so a crashed compile reports MISSING
// rather than 0:
//
//     origin                              tied interface     programs  min  max
//     scan/collectRuntimeResourceGroups    (memref<2xi64>)        294   21   21
//     other                                (memref<2xi64>)        126    1   77
//
// **Both recorded figures are right, and both are per-program.** The nameless
// bare-range scan is a hard constant at 21 -- identical in all 294, including a
// three-line `str` control that never mentions `int` -- and it is the part no
// declaration can reach. Everything above 21 comes from `other`, an origin
// NEITHER figure attributed, being the collectors that set no `OriginScope`. The
// per-program total ranges **21 to 98** (42 in 5 programs, 83 in 4, 98 in
// `w3_cross_random_json.py`). So the correct form is "21 invariant plus 0-77
// program-driven", and a single number quoted without its program is not
// checkable.
//
// I reproduced the 21 first, from three small probes, and concluded that ALL
// width-2 ambiguity came from the nameless scanner. **That was wrong, and the
// broader set refuted it**: `other` fires in 126 of 294 programs and my probes
// reached none of them. Recording it because the failure mode is the one this
// file keeps hitting -- a program-limited measurement that looks like a tree
// property.
//
// ⛔ UNRECORDED TIES FOUND BY THE SAME SWEEP, at widths this file lists as clean:
//
//     scan/...  (memref<16xi64>)                      10 programs   4-63
//     scan/...  (memref<16xi64>, memref<16xi64>)       6 programs   9-90
//     scan/...  (memref<8xi64>)                        7 programs   9-36
//     scan/...  (memref<3xi64>)                       21 programs   9-153
//
// `memref<16xi64>` is listed above as unique to `builtins.object`, so an
// ambiguous exit there needs a second candidate -- class instances taking
// object's shape is the obvious reading and it is a READING, not a measurement.
// ✅ NOW MEASURED, and the cause is stronger than the reading was: every user
// class gets a SYNTHESISED `ly.runtime.deallocator` (`RuntimeABI.cpp:1276`), so
// `class_object_field_ops.py` has 12 candidates at `(memref<16xi64>)` and not
// two. Item 5 of the block at the end of this file; the consequence is that
// width 16 is unique only in the manifests.
// The width-3 tie reaches 153, an order above the 18 recorded from
// `list_methods.py`. **None of these is int's, and none is fixed by converting
// int**; they are logged here because the sweep that answered int's question
// answered theirs too, and an unrecorded tie reads as an absent one.
//
// So read the rows below as "a tie whose reachability is unmeasured, and which
// four programs failed to reach", not as a live defect. That makes the case for
// spending a scarce width on one of them weaker, not stronger:
//
//     memref<3xi64>  LyBool, LyFloat, LyReadyIntAwaitable          (3)
//                    -- REACHED: 18 ambiguous exits on golden/cases/list_methods
//     memref<5xi64>  LyCoroutine, LyRange, LyRangeIterator         (3)
//     memref<2xi64>  LyLong, LyNullContext, LyUnicode              (3)
//                    -- was 4; LyBytes left for width 6 in `00079ef`
//     memref<4xi64>  LyAsyncCounter, LyCounter, LyReadyAsyncCounter (3)
//     memref<8xi64>  LyBytesIO, LyDict, LyEventLoop, LyFileIO,
//                    LyFunction, LyStringIO, LyTextIO              (7)
//     memref<6xi64>  LyBytes                                       (1)  clean
//     memref<7xi64>  LyComplex                                     (1)  clean
//     memref<9xi64>  LyList                                        (1)  clean
//     memref<11xi64> LySet                                         (1)  clean
//     memref<13xi64> LyFrozenSet                                   (1)  clean
//     memref<10xi64> LyFuture   memref<16xi64> LyObject   memref<64xi64> LyGenerator
//
// ===========================================================================
// MEASURED by the set/frozenset conversion, and it settles the three-type row
// above rather than leaving it as an unresolved ledger entry. Same instrumented
// C++ in both columns (the counter that produced the 63 above, re-applied and
// then reverted); ONLY `runtime/modules/builtins.mlir` differs. Nine programs,
// one per contract, main `7822be4` in the left column:
//
//                                                    main 7822be4   set@11 fset@13
//     ambiguous on (2xi64, 2xi64, ?xi64)                     63            0
//     ambiguous on (memref<2xi64>)                           56           21
//                                                           ---          ---
//     total ambiguous exits per compile                      119           21
//
//     resolved builtins.tuple                          (absent)           56
//     resolved builtins.set                            (absent)           14
//     resolved builtins.frozenset                      (absent)      (absent)
//     resolved builtins.list / str / int / bytes    56/1241/315/21  unchanged
//     contractName.empty()                                     0            0
//
// **The three-type tie is gone, 63 -> 0**, and it is gone structurally rather
// than by luck: with `list`, `set` and `frozenset` all off that interface,
// `builtins.tuple` is its sole owner, so the selection loop can no longer
// record a tie there at all. The reciprocal number is the informative one --
// `builtins.tuple` appears under `resolved` for the FIRST time, at 56. So the
// 63 were tuple traffic that the tie was denying, which is what the recorded
// "63 is invariant after `list` converted, therefore it is tuple/set/frozenset
// traffic" predicted.
//
// The independent cross-check: 63 + 56 = **119**, which reproduces exactly the
// 119 recorded above from a DIFFERENT instrument, a different agent and an
// earlier tree. Two instruments agreeing on the split is what makes either
// usable.
//
// NOT PREDICTED, and reported because it was not: `ambiguous on
// (memref<2xi64>)` also fell, **56 -> 21**. A manifest-only change to `set` and
// `frozenset` was not expected to move the width-2 figure at all. The likely
// mechanism is that `collectRuntimeResourceGroups` advances `++offset` on a
// failed match, so a three-value group whose first AND second lane types are
// both `memref<2xi64>` also feeds the arity-1 probe -- removing two such groups
// removes those probes. **That mechanism is read, not measured.** What is
// measured is only that the figure responded to the manifest change.
//
// Rule check, run before quoting any of the above: every figure in the table is
// invariant across all nine programs INCLUDING the `x = 1; y = x + 2` control
// that never mentions a set, so none of them is a measurement of the program --
// they are constants of lowering the runtime library. `nomatch` (3438-3688) and
// the `:450` fallback (1078-1096) do move, which is what shows the counter
// reads the input at all.
// ===========================================================================
//
// `builtins.float` at 3 and `builtins.range`/`range_iterator` at 5 sit in ties
// whose pre-existing owners (`LyBool`, `LyReadyIntAwaitable`, `LyCoroutine`) were
// there before those conversions, and which the 7.14 measurement did not reach.
//
// Why they are NOT being fixed by moving the width: the arithmetic at the top of
// this file. Three more moves would consume half the remaining free widths to
// protect three contracts, leave the other ties untouched, and add dead words to
// float and range for identity alone -- while `list`/`set`/`frozenset`/`tuple`,
// `int`, `str`, the exception family and class instances still have to fit. The
// fix for these is GAP 1 and GAP 2, not a width.
//
// `range` and `range_iterator` additionally tie with EACH OTHER by construction:
// same layout, same width, same one-lane shape. If a bare 5-word value can be
// either, no single width assignment for the pair helps -- it would take two
// distinct widths for two contracts that are deliberately structurally
// identical, which the arithmetic says is not affordable. This pair is the
// clearest demonstration that uniqueness is the wrong axis.
//
// `builtins.complex` at 7 is the one clean case, and it stays at 7 because it is
// already paid for; it is not evidence that the scheme works.
// ---------------------------------------------------------------------------

// EMPTY. The resource is exhausted, and this is the entry that records it.
//
// Provenance of each departure: `bytes` took 6 in `00079ef`, `complex` 7,
// `list` 9 in `7822be4`, `set` 11 and `frozenset` 13 here, and `tuple` 14 /
// `int` 15 are assigned on concurrent tracks (assigned BEFORE those branches
// merge, which is the whole reason this file exists -- grepping one tree for
// absence cannot see an unmerged branch, and that is how `complex` collided
// with `bytes` twice).
//
// So `builtins.str`, ~70 exception contracts and class instances have to convert
// with no unique width available. See the block at the top of this file: for
// them the two GAPs are not the durable alternative to a width, they are the
// only option. `str` is the one that most needed one -- playbook 8.3 measured 27
// sites depending on the `shapeMatch` score that one-laning `str` destroys.
//
// A converter must still update this list when a width is taken, because an
// EMPTY free list is information: it is what tells the next reader not to spend
// an afternoon looking for a free width before concluding there is none.
//
// Widths that are NOT free despite having no one-lane owner, with WHICH hazard
// each one carries -- (A) ambiguity, no group forms; (B) preemption, wrong
// deallocator selected; see above:
//
//     2   (A) 7-way single-input tie, and (B) leads the one remaining
//             multi-input interface of this family (builtins.tuple) plus
//             str_iterator. It led four before `list`, `set` and `frozenset`
//             left; `tuple` is the last.
//     4   (A) 3-way single-input tie (the lyrt counters)
//     8   (A) 7-way single-input tie -- every member already one-lane
//    12   (B) only: leading input of _asyncio.Task's and TaskIter's interfaces,
//             never a single-input interface of its own. `builtins.frozenset`
//             took 13 rather than 12 for exactly this reason -- 12 was assigned
//             away from this conversion as a pre-emption hazard, and (B) is the
//             mechanism that fails SILENTLY.
//
// Widths carrying (B) alone are the ones to distrust most, because (B) is the
// unmeasured mechanism AND it fails silently rather than by omission.
//
// Do not read this list as "pick one of these and you are safe". It is what is
// left of a resource that is NOW EXHAUSTED, kept accurate so the exhaustion is
// visible. A converter without a free width is not blocked, because the ties are
// already the tree's normal condition (width 8 ships with `dict` in a 7-way
// tie). Record the tie here and move on.
// EMPTY. 11, 13, 14 and 15 were the last four and are all assigned (set and
// frozenset CONVERTED, tuple CONVERTED, int reserved). See the boxed note above
// the assignment table: `str`, the exception family and class instances have no
// free width left, and for `str` that is not a cosmetic problem (§8.3).
//
// Spelled as a count rather than as `int kFreeHandleWidths[] = {}`, because a
// zero-length array is not valid C++ and this file should stay compilable if
// anyone ever does include it. (The set/frozenset track wrote the empty-array
// form and the tuple track wrote this one; the tuple track's is correct.)
inline constexpr int kFreeHandleWidthCount = 0;

// The scarcity a converter should see before assuming a width is available,
// and -- more importantly -- the ties that are ALREADY suppressing groups by
// mechanism (A) today, before any further conversion:
//
//   (memref<2xi64>, memref<2xi64>, memref<?xi64>)  **CLOSED: 63 -> 0.**
//       Was builtins.{list,set,frozenset,tuple} tied on interface AND shape,
//       four ways: the str/bytes configuration repeated. `list` left for width 9
//       and the count stayed 63, which is what identified the traffic as
//       tuple/set/frozenset's; `set` left for 11 and `frozenset` for 13, and the
//       count is now **0**, with `builtins.tuple` appearing under `resolved` for
//       the first time at 56. `tuple` is the sole owner of this interface now, so
//       the loop cannot record a tie here at all -- the row is closed rather than
//       reduced. Measured table above.
//
//       The `tuple` track recorded this as outstanding rather than skipped: "the
//       instrument is a counter on the contract-less `return nullptr` in
//       `common/Ownership.cpp`, and that file was explicitly out of scope for the
//       tuple/int track ... whoever lands `set` or `frozenset` can take it, and
//       the prediction to check is that the number falls."
//
//       **The set/frozenset track took it, and the prediction held: 63 -> 0.**
//       STATE THE TREE WITH THE NUMBER, because the two conversions were measured
//       on different ones. That measurement was taken on main `7822be4` -- `list`
//       converted, `tuple` NOT -- so it is the A/B of removing `set` and
//       `frozenset` from a THREE-way tie, and `builtins.tuple` appearing under
//       `resolved` at 56 is precisely tuple becoming the interface's sole owner.
//       On the merged tree `tuple` is at 14 as well, so the interface has NO
//       one-lane owner at all and the row is closed twice over, for two
//       independent reasons. Nobody has re-run the counter on the merged tree;
//       the algebra says it cannot record a tie with fewer than two candidates,
//       and that is a reading, not a measurement.
//
//       What breaking it surfaced, for the next converter's budget: nothing of
//       the str/bytes kind, and that was predicted by an audit rather than
//       discovered by a crash. Before the conversion, all 136 declarations
//       naming the three-type sequence were checked as a SET: 75 declare
//       `owned_results` and all 75 read exactly `[0]`, 10 declare
//       `transfer_args` and all 10 read `[0]`, 4 declare `release_args = [0]`,
//       57 declare nothing, and 0 declare a second owned root. The exception
//       family's `[0, 1]` over-declaration has no analogue here, so the tie was
//       hiding an INCOMPLETENESS (a group that never forms) rather than an
//       UNSOUNDNESS (a release that would double-free) -- and only the second
//       kind ships under `--release`. Mechanism (C) was clear too: of the 67
//       declarations whose owned result 0 IS that group, 46 name a result
//       contract, 13 default to a contract that is itself a group member (so the
//       default is correct), 8 are contract-less private helpers, and 0 name a
//       foreign contract without naming the result. `list` had no `@LyLong_Repr`.
//
//       What breaking it surfaced for `set`/`frozenset`, and it is NOT of the
//       same kind: the audit above was already done as a SET over all four
//       members, so it covered these two, and nothing of the str/bytes kind was
//       waiting. But the conversion did fix a **silent use-after-free** that no
//       audit of ownership DECLARATIONS could have found, because it was not a
//       declaration defect at all -- it was the transfer contract itself. On main
//       `7822be4`, `before = u; u.update(big)` where the update crosses the
//       64-slot initial capacity leaves `before` naming the pre-growth `items`
//       lane, which `ensure_capacity` has freed. It compiles clean, exits 0, and
//       prints a wrong membership answer; under libgmalloc it SIGSEGVs in both
//       guard orders, and under `MallocScribble` it prints a different wrong
//       answer, which is how the freed read is distinguished from a logic bug.
//       That is the `--release`-surviving shape, so the tie was hiding an
//       unsoundness here after all -- just not one reachable from the attribute
//       census. `tests/golden/cases/set_one_lane_interior.py` pins it.
//   (memref<8xi64>)   7 contracts -- and builtins.dict IS ALREADY MERGED here.
//       The same reasoning applies with no conversion pending, so any
//       suppression at this width is live in main right now. Highest-priority
//       measurement of the set; assigned separately. The cheap instrument is the
//       one that produced 28-vs-0: count the contract-less `return nullptr` and
//       the structural resolutions for `builtins.dict` over a few lines of dict
//       code, then compare against dict moved to a free width.
//   (memref<2xi64>)   8 contracts
//   (memref<4xi64>)   3 lyrt counters
//
// Adding a width to the free list requires checking BOTH: that no
// `ly.runtime.shape` declares it as a single lane, and that no
// `ly.runtime.deallocator` takes it as a single input or as the leading input
// of a multi-input interface.
//
// ---------------------------------------------------------------------------
// Mechanism-(C) blockers: declarations to fix BEFORE the named conversion.
// These are not width problems -- they are missing `ly.runtime.result_contract`
// declarations that a width coincidence currently makes survivable, and the
// coincidence ends when the width moves. Listed here because the width move is
// what exposes them.
//
// ✅ RESOLVED (2026-07-28): the `@LyLong_Repr` entry below is DONE, and the
// mechanism-(C) repair was not the attribute alone. `5c52ae0` (merge `522917d`)
// both declared `ly.runtime.result_contract = "builtins.str"` on `@LyLong_Repr`
// and made site A (`collectContractOwnedResultGroups`) prefer the declared name
// over the receiver's -- which is the half that made the attribute do anything,
// because the attribute alone was measured byte-identical over 10 programs while
// the receiver name was still preferred. 24 owned results were retargeted.
// `DriverTest.IntReprStringIsReleasedByStrDeallocator` pins it. Verified by
// reading `builtins.mlir` on `bcfbbf9`: the attribute is present.
//
// Keeping the entry rather than deleting it because the ORDER matters to the next
// converter: an attribute whose reader prefers a different name is inert, so
// "declare the result contract" is not by itself a fix for (C).
//
//   before `builtins.str`:  @LyLong_Repr (`builtins.mlir:8614`) -- DONE, above
//       `contract = "builtins.int"`, `__repr__`, owned result 0 is a
//       `builtins.str` header pair, and it declares NO result contract --
//       `@LyLong_Str` immediately above it does, which is why the omission is
//       invisible. Identical in shape to the `@LyFloat_Repr` defect this branch
//       fixed. It is latent only because `LyLong_DecRef`, `LyUnicode_DecRef` and
//       `LyFloat_DecRef` have byte-identical bodies (cast,
//       `LyObject_ReleaseStorageToZero`, conditional `memref.dealloc`), so the
//       wrong deallocator does the right thing. Converting `str` breaks that
//       identity. `int` was outside the float/complex/range scope, so this is
//       recorded rather than fixed: adding the attribute re-attributes the
//       result's resource and can move release placement, which needs a build
//       and a full suite to land safely.
//
//   before `builtins.int`:  @__ly_boxed_long_view
//       Reads box pointer words 5 and 6, i.e. int's LANE 1 and LANE 2 addresses,
//       which one-laning leaves uninitialised. The last lane-indexed raw box
//       accessor in the tree. Three callers, so equality, ordering and float
//       coercion all route through it. Invisible to every static oracle -- see
//       the playbook's blocker note for the full statement.
//
//       STATUS after the tuple/int track: **still unfixed, and NOT yet
//       observed to fire.** `int` was not converted (see the reservation note
//       below), so the words it reads are still the live lane addresses and the
//       accessor is still correct. What this track can add is a negative
//       result worth having: the mixed numeric tower was exercised hard
//       THROUGH one-lane tuple slots -- `(1.0,) == (1,)`, `(True,) == (1,)`,
//       `(1, 2.0, True) == (1.0, 2, 1)`, `sorted([(2.0,), (True,), (1,)])`,
//       `hash((1,)) == hash((1.0,))`, tuple keys mixing int and float -- all
//       byte-identical to CPython 3.14 with exit 0 and no SIGSEGV.
//       **So converting the CONTAINER a boxed int sits in does not disturb
//       this accessor; only converting `int` itself will.** That is the
//       expected reading of the mechanism rather than a surprise, but it had
//       not been checked, and it means the tuple conversion is not evidence
//       either way about the blocker. The fix (route through box word 2, the
//       ENTITY word, then the handle's own words) is unchanged and still has
//       to land in the same change that flips `LyLong_Shape`.
//
//       WHY IT CANNOT LAND EARLIER, checked 2026-07-28 because "de-risk the
//       blocker first" is the obvious plan and it does not work. The fix routes
//       through the entity word and then reads sign/count/digits out of **the
//       handle's own words** -- which only exist once int is one-laned. While int
//       is three lanes the entity word yields the header alone, and sign, count
//       and digits base are reachable only through the lane addresses in box
//       words 5 and 6, i.e. exactly what the accessor already does. So the
//       accessor is CORRECT today and the repair is genuinely coupled to the
//       shape flip; there is no separable preparatory commit. Verified against
//       `builtins.mlir:15530` on `bcfbbf9`: still `%c5`/`%c6`, three callers
//       (`__ly_box_equal_numeric`, `__ly_boxed_num_as_f64`, `__ly_box_less`).
//
// Verified clear for float/complex/range on this branch (the check to repeat per
// conversion): of the functions whose owned result 0 is a ONE-LANE handle at a
// width these contracts own (3, 5, 7), 38 declare their own contract, 6 declare
// a foreign contract but NAME the result -- `LyLong_Float`, `LyLong_TrueDiv`,
// `LyComplex_Abs`, `LyUnicodeData_Numeric`, `LyTask_GetCoro`,
// `LyReadyAsyncCounter_ANext` -- and 0 declare a foreign contract without naming
// the result. `LyLong_TrueDiv` was the sole exception and this branch fixed it,
// which turns out to have been necessary for the conversion's own safety rather
// than an adjacent tidy-up.
//
// A warning about how to run that check: filter on the result list having length
// ONE. A first attempt matched a LEADING `memref<3xi64>` and reported 157 hits,
// almost all of them the exception triple `(memref<3xi64>, memref<2xi64>,
// memref<?xi8>)` whose header is also 3 words -- i.e. the check itself committed
// the "width is not a proof of kind" fallacy this file exists to prevent. And
// even with the right filter the result is a CANDIDATE list, not an answer:
// widths 3 and 5 have several one-lane owners (3: float, boxed bool,
// lyrt.ReadyIntAwaitable; 5: range, range_iterator, types.CoroutineType), so
// which contract a length-1 `memref<3xi64>` result actually is cannot be decided
// from types. Only the declaration decides -- which is the point.

// ===========================================================================
// ⛔ `builtins.int` AT 15 IS DEFERRED, AND THE REASON IS NOT THE WIDTH.
// Measured 2026-07-28 on main `bcfbbf9`, Debug, load recorded per run below.
// The section after this one describes the conversion and is still accurate
// about HOW; this block is the adjudication of WHETHER, and it says not now.
//
// 1. WHAT TAKING WIDTH 15 GUARANTEES. Not "uniqueness" -- that framing is what
//    the census refuted, since 16 of 28 single-input deallocators share a width
//    and sharing is the tree's normal condition. What 15 buys is narrower and
//    checkable: **exact-type SOLE CANDIDACY**. `valueRangeMatchesTypes` compares
//    types by equality (`common/Ownership.cpp:388`), and no deallocator interface
//    in any manifest mentions `memref<15xi64>` in ANY position -- verified against
//    both claim shapes, single-input and every position of a multi-input
//    interface, and now machine-checked by ctest
//    `abi.handle_width_reservations`. So for a value range where int's release
//    interface type-matches, the selection loop finds exactly one candidate:
//    `ambiguous` cannot be set, and the shape score is never consulted.
//
//    That is STRICTLY STRONGER than the `shapeMatch` = 3 it replaces, which only
//    wins where int's full three-lane shape matches at that exact offset and
//    which loses outright to any equal-arity rival that also scores 3.
//
//    What it does NOT guarantee, explicitly: nothing about the other four ties,
//    nothing about `declared_name_absent` (308, invariant), and nothing about
//    mechanism (C) -- except that a wrong declared name would begin failing to
//    type-check and fall to `:450`, which is the benign direction.
//
// 2. THE PROTECTION IS 110 SITES, NOT 27, AND IT TRANSFERS INTACT. §8.3's "27
//    sites" is `str`'s figure; int's is larger because int scores 3. Replaying
//    `collectRuntimeResourceGroups` over the manifests with its real offset
//    advance (`offset += inputTypes + views`, so a group that covers an offset
//    hides it) gives, over 1180 functions with results and 37 deallocators:
//
//                                    BEFORE      int@15      int@2 (control)
//      total ambiguous exits            80          80            190
//      ambiguous on (memref<2xi64>)      4           4            114
//      resolved builtins.int           110         110            110
//      int resolutions decided by
//        shapeMatch                    110           0              0
//
//    **All 110 of int's structural resolutions depend on the shape score, and
//    all 110 survive the move to 15 as sole-candidate resolutions.** The control
//    column is the one that earns the width: one-laning int WITHOUT moving it off
//    2 costs +110 ambiguous exits, and 114 - 4 = 110 closes the arithmetic
//    exactly against the shape-decided count. The width choice is therefore
//    load-bearing and correct -- 12 is a mechanism-(B) hazard and 13/14 are
//    taken, so 15 is the narrowest safe free width.
//
// 3. AND THE CONVERSION REDUCES AMBIGUITY BY ZERO. This is the finding that
//    changes the schedule. 80 -> 80 total, 4 -> 4 at width 2. Removing int from
//    the width-2 tie leaves `str` and `nullcontext` still tied there, and a
//    two-way tie returns nullptr exactly as a three-way one does. So the
//    conversion is not an ambiguity fix, and it cannot be justified as one.
//
// 4. NOR IS THERE A DEFECT OF THE set/frozenset KIND. That one needed a mutable
//    interior buffer held as a separate SSA lane, so a holder could keep a lane a
//    reallocation freed. Checked for int by a route independent of the shape
//    audit above -- enumerating allocation sites in the int family -- there are
//    **two, both fresh-entity** (`__ly_long_alloc_raw`, `LyLong_FromI64`) and
//    **zero reallocation sites**. int entities are immutable once normalised and
//    arithmetic allocates a new one, so no int digits lane can go stale. The
//    scratch buffers in the format/divmod paths are function-local and freed in
//    the same body.
//
// 5. WHAT IT COSTS, which is the other half of the trade. A width IS object size,
//    and for int the multiplier is the worst in the tree. The block is
//    `32 + 4*capacity` bytes today (header 16, meta 16, digits inline); at width
//    15 the handle view alone forces 120, giving `120 + 4*capacity`. A
//    single-limb int goes from ~48 B to ~128 B allocated -- **+80 B, 2.67x** --
//    and 10 of those 15 words are dead space bought for identity. This is not a
//    cold path: the user body of a four-line `for i in range(10): n = n + i`
//    program references `LyLong_FromI64`/`__ly_long_alloc` 10 times and calls
//    `LyLong_DecRef` 7 times, so ordinary integer code allocates int entities.
//
// 6. AND THE MECHANICAL SCOPE REPRODUCES EXACTLY. The §8.3 experiment re-run on
//    `bcfbbf9` (flip `LyLong_Shape` alone, build, attribute, revert): **107 gate
//    errors over 107 distinct declarations in 9 manifests** -- builtins 65, _io
//    17, posix 7, _time 5, _random 3, lyrt 3, math 3, unicodedata 2, asyncio 2;
//    by check, 65 declared result, 39 method receiver, 2 next element, 1
//    initializer. Identical to the figures recorded below, which are therefore
//    confirmed rather than stale. Below the declaration level, 455 lines across
//    10 manifests name int's triple and the family is 95 functions.
//
// SO: the conversion is SAFE on the width axis and buys no measured safety today,
// against a 2.67x size regression on the language's hottest type, a ten-manifest
// change, and the `@__ly_boxed_long_view` blocker below whose failure mode is
// "prints correct output, then SIGSEGVs" and which no static oracle sees. The
// registry's own argument applies to itself here: a contract at a wider width
// "just delays the collision while paying for it on every allocation."
//
// WHAT WOULD CHANGE THIS, so the next reader does not re-litigate it:
//   * a program that reaches int's structural path and mis-attributes -- the
//     instrument is `LYTHON_DEALLOC_CENSUS=1` and the key to watch is a
//     `resolved builtins.int` that MOVES when int is not in the program;
//   * `Interior` conformance being required rather than desirable for int (it is
//     9/13 today and int's observable consequence is satisfied);
//   * the size cost being retired by an unboxed small-int representation, after
//     which 15 words costs nothing on the common path. **This is the one that
//     makes the conversion cheap, and it is the sequencing recommendation:
//     unbox small ints first, convert second.**
//
// NOT MEASURED, and stated as such: whether the conversion would fire
// `dropUnusedLogicalBlockArguments` (`ABI/Returns.cpp`). Its index defect was
// found and fixed by reading during this work -- it used a block-argument index
// on an already-shortened branch operand list -- but no program in the suite
// builds a block that reaches it, so "could not construct a reaching form" is
// what is established, not "safe".
// ===========================================================================
//
// ===========================================================================
// ⛔ AND SO ARE THE OTHER THREE. `builtins.str`, the exception family and class
// instances measured 2026-07-28 on main `51fb04d` by the same replay, which is
// now a file (`tests/probe/tools/laneswap.py`) rather than a number in a commit
// message -- it reproduces int's published 80 / 80 / 190, 4 / 4 / 114 and
// 110 / 110 columns exactly before answering anything new.
//
// This retires the item this whole file is organised around. The header opens
// with "Still to convert: list, set, frozenset, tuple, int, str, ~70 exception
// contracts, and class instances" and treats the empty free list as the crisis.
// Measured, the tail of that list is TWO conversions and one non-item, and both
// conversions reduce ambiguity by exactly zero:
//
//                                      BEFORE   at a free width   at width 2/3
//   total ambiguous exits                 80          80            331 / 312
//   ambiguous on (memref<2xi64>)           4           4            255 /   4
//   ambiguous on (memref<3xi64>)          35          35             35 / 267
//   resolved builtins.str                251         251                251
//     of which decided by shapeMatch     251           0                  0
//   resolved builtins.BaseException      232         232                232
//     of which decided by shapeMatch       0           0                  0
//
//   (`str` columns are str@17 and str@2; exception columns are
//    BaseException@18 and BaseException@3. 17 and 18 are hypotheticals -- there
//    is no free width -- chosen because a width no interface mentions is what
//    sole candidacy needs. 331 - 80 = 251 and 255 - 4 = 251 close against str's
//    shape-decided count; 312 - 80 = 232 and 267 - 35 = 232 close against the
//    exception family's arity-decided count. The controls earn their widths
//    exactly as int's did, and the widths still buy nothing.)
//
// 1. `str` IS THE SAME ANSWER AS `int`, FOR THE SAME REASON. 80 -> 80, 4 -> 4.
//    Removing `str` from the width-2 tie leaves `int` and `nullcontext` still
//    tied there, and a two-way tie returns nullptr exactly as a three-way one
//    does. The protection at stake is 251 sites (§8.3's "27" is not the figure,
//    the same way it was not int's 110), and all 251 survive a free width as
//    sole-candidate resolutions -- so the conversion is safe on the width axis
//    and, again, is not an ambiguity fix.
//
// 2. BUT THE WIDTH-2 TIE IS NOT A THREE-STEP LADDER, AND EMPTYING IT IS WORSE
//    THAN LEAVING IT. This is the finding that generalises past `str`, and it
//    contradicts the obvious reading of the rows above. Moving BOTH `str` and
//    `int` off 2 does clear it -- 4 -> 0, total 80 -> 76, and only two of the
//    three have to move because `contextlib.nullcontext` declares no
//    `ly.runtime.shape` and becomes the sole candidate. But look at WHICH four
//    exits those are, and what they become:
//
//        __ly_long_operand_view   (memref<2xi64>, memref<?xi32>)   off 0
//        __ly_str_iterator_alloc  (memref<2xi64>, memref<2xi64>)   off 0
//        __ly_str_iterator_alloc  (memref<2xi64>, memref<2xi64>)   off 1
//        LyNullContext_New        (memref<2xi64>)                  off 0
//
//    Three of the four are an int meta+digits view and a str iterator's header
//    and state. Neither is short by accident: `LyUnicodeStrIterator_DecRef` is a
//    FOUR-input interface (iterator header, state, source header, source bytes)
//    and `__ly_str_iterator_alloc` returns only the first two, so the scan is
//    probing a range that no deallocator can cover. With `str` and `int` gone
//    those offsets do not become clean -- all four become
//    **`contextlib.nullcontext` resolutions** (measured, not derived: 4 sites,
//    the same 4), because sole candidacy does not check that the value IS a
//    nullcontext, only that the types match. That trades mechanism (A), which
//    drops a group and presents as a leak, for mechanism (C), which returns
//    `LyNullContext_DecRef` for an int's digits view and presents as nothing at
//    all. **Emptying a tie is only a fix when the survivor is the right answer
//    at every offset it covered**, and at width 2 it is not. Anyone proposing
//    to move `str` or `int` off 2 "to clean up the tie" is proposing this, and
//    must give `nullcontext` a shape or a width of its own FIRST.
//
// 3. NOR IS THERE A set/frozenset DEFECT IN `str` -- and it holds a real lane,
//    so this is the case the audit said to check rather than assume. `str`'s
//    canonical shape has a genuine second SSA lane (`memref<?xi8>`), unlike
//    every single-lane member of the width 3/4/5/8 ties. Counted rather than
//    read (`lanemutate.py`, whose entity-allocation criterion reproduces int's
//    recorded "two, both fresh-entity" -- `__ly_long_alloc_raw` 5874 and
//    `LyLong_FromI64` 6181 -- before being pointed at str):
//
//        entity allocation sites for builtins.str .......... 1  (fresh-entity)
//          __ly_unicode_alloc, builtins.mlir:8820
//        reallocation sites ............................... 0
//        sites that free a str block ...................... 1
//          LyUnicode_DecRef, builtins.mlir:2073
//
//    The structural reason the count is 0: the bytes lane is
//    `memref.view %block[24]` into the entity's own single allocation, so its
//    address is a function of the header's and there is no stored base-address
//    word that a reallocation could overwrite. Only FOUR `memref.dealloc` of a
//    `memref<2xi64>` exist in all 19 manifests.
//
//    The nearest counterexample, checked and rejected rather than not found:
//    `LyBuiltin_Input` (builtins.mlir:16651, 16668) really does grow a byte
//    buffer by allocate-copy-free. The buffer is function-local and is never a
//    `builtins.str` lane -- no str exists until `LyUnicode_FromBytes` at 16703,
//    after the growth loop, and the raw buffer is freed at 16704. A
//    reallocating buffer that is not a lane is not the defect.
//
// 4. THE EXCEPTION FAMILY IS ONE CONTRACT, NOT ~70. The "~70 exception
//    contracts" in this file's opening arithmetic -- half of the 24-vs-6 crisis
//    that declared the free list exhausted -- is a count of NAMES, and the
//    thing a conversion would touch is one declaration:
//
//        contracts naming ly.runtime.contract anywhere ........ 104
//          of which declare a deallocator ..................... 34
//        exception-shaped contracts .......................... 67
//          of which declare a deallocator ..................... 1   BaseException
//          of which declare a shape ........................... 66
//          distinct shapes among those 66 ..................... 1
//              (memref<3xi64>, memref<2xi64>, memref<?xi8>)
//
//    And the 66 shapes are INERT in the deallocator table: the shape walk in
//    `collectRuntimeDeallocators` joins by contract name, and a contract with
//    no deallocator entry has nothing to join to, so all 66 are dropped. The
//    table is 34 entries and the family contributes exactly one.
//
//    THE REPRESENTATION, checked separately from the inheritance, because a
//    shared deallocator does NOT prove a shared layout -- the dict-payload track
//    found "same code so same defect" wrong in both directions, and this is the
//    same shape of claim. Not the deallocator this time, the constructors:
//
//        per-subclass exception constructors (_New + _Init) ....... 134
//          that ALLOCATE their own block ......................... 0
//          that FORWARD to LyBaseException_{New,Init} ............ 132
//          the family constructors themselves .................... 2
//        distinct return type lists among all 134 ................. 1
//
//    plus the independent entity-allocation census, which finds ONE allocation
//    site in the family (`LyBaseException_New`, builtins.mlir:2284) taking
//    `%class_id: i64 {ly.runtime.class_id_argument}` -- the subclass is DATA in
//    a word of the block, not a layout. Three routes, one representation. Worth
//    naming the case that would have broken it: CPython's `UnicodeDecodeError`
//    carries encoding/object/start/end/reason, and Lython's does not -- its
//    every signature is the same triple, so the one subclass with a reason to
//    diverge does not.
//
//    `LyBaseException_DecRef` is also not one-lane-able in the sense the rest of
//    this file uses. It is a THREE-input interface with no shape declaration, so
//    its shapeTypes default to its inputTypes and its shape score is 0 always.
//    All 232 of its resolutions are won by ARITY -- the mechanism (B) path this
//    file documents, and the 206 "inherited by design" preemptions
//    `preemption.py` reports are the same fact seen from the other side.
//    Converting it to one lane at a free width preserves all 232 as
//    sole-candidate resolutions and changes ambiguity by nothing; converting it
//    at the entity-root width 3, which is what "one-lane it" naively means,
//    destroys all 232 into the bool/float/ReadyIntAwaitable tie.
//
// 5. CLASS INSTANCES ARE NOT A PENDING CONVERSION, AND UNIQUENESS IS NOT MERELY
//    EXHAUSTED FOR THEM -- IT IS UNDEFINABLE. The unrecorded `memref<16xi64>`
//    ties logged above needed "a second candidate", and the note says class
//    instances taking object's shape is a READING. Measured, the cause is
//    stronger than the reading: `RuntimeABI.cpp:1276` SYNTHESISES a
//    `ly.runtime.deallocator` per user class, named `__ly_dealloc_<Class>`, with
//    its own contract name, an interface built out of `memref<16xi64>` object
//    handles, and NO `ly.runtime.shape` -- so score 0. Dumping
//    `golden/cases/class_object_field_ops.py`:
//
//        (memref<16xi64>)  12 candidates -- LyObject_DecRef plus __ly_dealloc_
//          {Box,DictBox,DictHolder,Factory,Four,FourHolder,Holder,Inner,
//           ListBox,ListHolder,SelfStore}
//
//    and `cross_nested_field_chain.py` gives `__ly_dealloc_{L1,L2,L3}` all at
//    `(memref<16xi64>, memref<16xi64>)`, a three-way arity-2 tie, which is the
//    `(16, 16)` row. So width 16 is "unique to builtins.object" only in the
//    manifests; in a compiled program the candidate set is generated from the
//    user's source, and N classes of equal field arity give N+1 candidates.
//
//    PRECISELY, because "class instances are one-lane" is nearly right and the
//    near-miss matters: the HANDLE is always `memref<16xi64>`, so no width is
//    ever chosen for a class. The ARITY is not fixed -- it is one plus the
//    storage lanes of the fields, so `Inner` is `(16)` and `L1` is `(16, 16)`.
//    A class whose interface is already `(16)` is therefore in an (N+1)-way tie
//    with every other such class in the same program, and one-laning the
//    multi-lane ones would move them INTO that tie rather than out of anything.
//    That is int@2 structurally -- a conversion that strictly increases
//    ambiguity -- and it is DERIVED from the arity rule here, not measured,
//    because the arm cannot be replayed against the manifests: the deallocators
//    do not exist until a user program is compiled.
//
//    **No entry in a checked-in registry can separate two user classes**: both
//    are `memref<16xi64>` by construction, and the program that collides them
//    does not exist when the registry is written. This is the strongest form of
//    the argument the `range`/`range_iterator` row already makes -- a tie no
//    width can break -- and it is the case that ends the scheme rather than
//    stressing it. The fix is GAP 1 and GAP 2; the names here already exist, and
//    the sweep confirms the surviving traffic is the nameless origin.
//
// 6. SUITE-WIDE, and this reproduces the unrecorded rows above with a cause
//    instead of a reading. `LYTHON_DEALLOC_CENSUS=1` over 292 golden cases, rc
//    recorded per program, 0 MISSING:
//
//     origin                              tied interface          progs min max
//     scan/collectRuntimeResourceGroups   (memref<2xi64>)           289  21  21
//     other                               (memref<2xi64>)           126   1  77
//     scan/...                            (memref<3xi64>)            20   9 153
//     scan/...                            (memref<16xi64>)           10   9  63
//     scan/...                            (memref<8xi64>)             7   9  36
//     scan/...                            (memref<16xi64>, 16xi64)    6   9  90
//     scan/...                            (memref<5xi64>)             1  18  18
//     other                               (memref<16xi64>)            1   4   4
//     other                               (memref<16xi64>, 16xi64)    1  10  10
//
//    The width-16 rows are class programs (`class_object_field_ops`, `union`,
//    `stdlib_time`, `stdlib_os_fs`, `w3_cross_os_try_rebind`, `stdlib_pathlib`,
//    `loop_call_object_args`, `class_object_field_store`, `generic_classes`,
//    `namedtuple_desugar`, `cross_nested_field_chain`,
//    `interior_view_nested_chain_grow`), i.e. exactly the surface in item 5.
//
// SO: three items, three times not now. `str` is int's answer with a worse
// multiplier -- its block is `24 + count*width` bytes, so a one-character
// latin-1 str at a 17-word handle goes 25 B -> 137 B, **5.5x**, against a
// measured ambiguity reduction of zero and a 548-signature surface. The
// exception family is one declaration whose conversion also changes nothing,
// and whose current protection is arity rather than shape. Class instances have
// no width to schedule at all -- their handle is `memref<16xi64>` by
// construction and their tie is generated per program.
//
// WHAT WOULD CHANGE THIS:
//   * for `str` and the exception family, the same triggers as int -- a program
//     that reaches the structural path and MIS-ATTRIBUTES
//     (`LYTHON_DEALLOC_CENSUS=1`, watch a `resolved builtins.str` that moves
//     when no str is in the program);
//   * for the width-2 tie specifically, `contextlib.nullcontext` getting a shape
//     or a width of its own -- until then item 2 says emptying that tie makes it
//     worse, and that is a REASON NOT TO CONVERT rather than a caveat;
//   * for class instances, nothing on this axis. Only GAP 1 / GAP 2.
//
// WHAT IS NOT ESTABLISHED, and is stated as such: all of the above is the STATIC
// surface plus per-program census counts. That the four width-2 exits and the
// width-16 exits DROP A RELEASE rather than being speculative probes that some
// named path already covers is NOT shown -- `ctest -L leak` is green on all five
// and no leaking program has been constructed. "Could not construct a reaching
// form" is what holds, not "safe".
// ===========================================================================
//
// ---------------------------------------------------------------------------
// `builtins.int` at 15: RESERVED, NOT CONVERTED. Why the width is booked before
// the work, and what the work actually measures.
//
// The width is recorded now because reserving it is the whole point of this
// file: 15 was one of the last four free single-input interfaces, and leaving
// it unclaimed while `int` is known to be next is how `complex` collided with
// `bytes` twice. Booking it costs nothing and closes that race.
//
// What stopped the conversion was SIZE, measured rather than estimated, by the
// §8.3 scoping experiment (flip `LyLong_Shape` alone, build, attribute the gate
// output by the OWNING contract of each failing declaration, revert):
//
//     107 gate errors over 107 distinct declarations, in NINE manifests
//       builtins.mlir 65   _io 17   posix 7   _time 5   _random 3
//       lyrt 3   math 3   unicodedata 2   asyncio 2
//     by owning contract: builtins.int 55, and 52 in SIXTEEN other contracts
//       (str 13, _io.{FileIO,BytesIO,StringIO,TextIOWrapper} 17, float 5,
//        bytes 3, object 3, tuple 2, list 2, _asyncio.Task 2, range 1,
//        range_iterator 1, lyrt.{ReadyIntAwaitable,AsyncCounter,Counter} 3)
//     by check: 65 declared result_contract, 39 method receiver,
//               2 next element result, 1 initializer result
//
// Compare `tuple`, taken the same way on the same tree: 21 errors over 19
// declarations, ALL in `builtins.mlir`, 16 of them tuple's own. That is the
// difference between a one-file change and a nine-file one, and it is the same
// 40/60-style split that got `str` rescheduled -- `int` is 51/49.
//
// The surface below the declaration level is larger again: 155 signatures name
// int's operand triple `(memref<2xi64>, memref<2xi64>, memref<?xi32>)` (98 in
// `builtins.mlir`, 57 outside it), and `memref<?xi32>` occurs 1206 times across
// the manifests. `int` sits between `float` (199 sites, converted) and `str`
// (548 signatures, deferred), and closer to `str`.
//
// THE DESIGN IS SETTLED, so this is a scheduling item and not an open question.
// `int` is neither §2's pattern (payload behind an address word, a second
// allocation) nor §7's (fixed-width inline payload). It is `bytes`'s: a
// VARIABLE-length payload inline at a fixed byte offset of the entity's single
// allocation. `__ly_long_alloc_raw` already lays out one block as [0,16) header,
// [16,32) meta, [32,..) digits. So:
//
//     word 0  refcount        word 3  digit count
//     word 1  class id (1)    word 4  digits base address
//     word 2  sign            words 5..14 pad
//     digits at byte offset 120, inside the same allocation
//
// -- which keeps ONE allocation and one `memref.dealloc`, exactly as
// `bytes_abi::kPayloadByteOffset` does at 48. Do NOT give int's digits their own
// allocation: it would double the malloc count on the language's hottest type.
// The `__ly_long_*` private helpers taking `(%meta, %digits)` views can keep
// their signatures; only the public boundaries derive the views from the handle.
//
// Two prerequisites, both inside `int`'s own change:
//   1. `@__ly_boxed_long_view` -- the mechanism-(C)/box-accessor blocker above.
//   2. `@LyLong_Repr` -- ✅ ALREADY DONE in `5c52ae0`, no longer a prerequisite.
//      The reasoning that put it here still holds and is worth keeping: once
//      `LyLong_DecRef` takes `memref<15xi64>` it stops type-checking against the
//      str header pair, so the single-candidate (C) path would stop resolving.
//      That is now moot because the declared name selects `builtins.str`
//      directly. Do NOT re-derive this as outstanding work.
// ---------------------------------------------------------------------------

} // namespace py::lowering::handle_width
