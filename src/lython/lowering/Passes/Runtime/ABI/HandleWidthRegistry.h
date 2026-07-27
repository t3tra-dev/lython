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
//         verifier/runtime/AffineOwnership.cpp:61   (returnCarriesGroupInsideOwnedAggregate)
//         common/Ownership.cpp:995                  (callableOwnedReturnRanges helper)
//         common/Ownership.cpp:1311                 (staticEvidenceCoveredLogicalOffsets)
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
//       NOT MEASURED -- this is read off the algebra, and this project has
//       repeatedly found that the algebra does not tell you whether the path
//       carries traffic (rule 8, rfc/stdlib-semantics.md). Recorded as a
//       mechanism with the measurement outstanding.
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
//      11  builtins.set                                     assigned to the
//                                                           set/frozenset track
//      12  _asyncio.Task                                    leading word of a
//                                                           multi-input release
//      13  builtins.frozenset                               assigned to the
//                                                           set/frozenset track
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
//                                                           CONVERTED -- see the
//                                                           note at the end of
//                                                           this file
//      16  builtins.object                                  payload box / boxed
//                                                           field slot
//      64  types.GeneratorType                              frame
//
// ===========================================================================
// THE FREE LIST IS NOW EMPTY. This is the arithmetic at the top of the file
// arriving, and it is worth stating as a fact rather than a projection.
//
// Widths 11, 13, 14 and 15 were the last four free single-input release
// interfaces. They are spoken for: `set`->11, `frozenset`->13, `tuple`->14,
// `int`->15. **Nothing is left for `builtins.str`, the ~70 exception contracts,
// or class instances** -- the three largest items remaining in stage 4b, and
// the one (`str`) whose conversion §8.3 measured as DEPENDING on a unique width
// (one-laning it drops its `shapeMatch` from 2 to 0 and sweeps 27 sites into
// the eight-way `memref<2xi64>` ambiguity).
//
// So the interim mitigation has run out before the work has. Those three must
// either land on a tied width and accept mechanism (A), or GAP 1 and GAP 2 must
// be closed first. **The scheme was always the contract name; this is the point
// at which there is no longer a local rescue to fall back on.** Do not read the
// empty free list as "pick a bigger number": width is object size, the ties are
// on the release interface rather than the shape, and a contract at width 20
// still scores 0 and still cannot win a tiebreak -- it just delays the
// collision while paying for it on every allocation.
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
//     memref<10xi64> LyFuture   memref<16xi64> LyObject   memref<64xi64> LyGenerator
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

// Free as of main `00079ef` (`bytes` merged at 6) plus this branch at 7.
// Verified against the merged tree by re-running the census after the rebase:
// `bytes` is the sole owner of 6, `complex` the sole owner of 7, and the
// `memref<2xi64>` group dropped 4 -> 3 as `bytes` left it. FIVE widths for the eight-plus
// contracts still to convert -- see the scaling note at the top before treating
// this list as a plan.
//
// Widths that are NOT free despite having no one-lane owner, with WHICH hazard
// each one carries -- (A) ambiguity, no group forms; (B) preemption, wrong
// deallocator selected; see above:
//
//     2   (A) 7-way single-input tie, and (B) leads three multi-input
//             interfaces: list/set/frozenset/tuple, and str_iterator
//     4   (A) 3-way single-input tie (the lyrt counters)
//     8   (A) 7-way single-input tie -- every member already one-lane
//    12   (B) only: leading input of _asyncio.Task's and TaskIter's interfaces,
//             never a single-input interface of its own
//
// Widths carrying (B) alone are the ones to distrust most, because (B) is the
// unmeasured mechanism AND it fails silently rather than by omission.
//
// Do not read this list as "pick one of these and you are safe". It is what is
// left of a resource that is nearly exhausted, kept accurate so the exhaustion
// is visible. A converter with a free width available may take one as a local
// mitigation; a converter without one is not blocked, because the ties are
// already the tree's normal condition (width 8 ships with `dict` in a 7-way
// tie). Record the tie here and move on.
// EMPTY. 11, 13, 14 and 15 were the last four and are all assigned (set,
// frozenset, tuple, int). See the boxed note above the assignment table: `str`,
// the exception family and class instances have no free width left, and for
// `str` that is not a cosmetic problem (§8.3).
//
// Spelled as a count rather than as `int kFreeHandleWidths[] = {}`, because a
// zero-length array is not valid C++ and this file should stay compilable if
// anyone ever does include it.
inline constexpr int kFreeHandleWidthCount = 0;

// The scarcity a converter should see before assuming a width is available,
// and -- more importantly -- the ties that are ALREADY suppressing groups by
// mechanism (A) today, before any further conversion:
//
//   (memref<2xi64>, memref<2xi64>, memref<?xi64>)  x63 ambiguous exits per
//       compile, MEASURED (see the correction above), invariant across ten
//       programs. Was builtins.{list,set,frozenset,tuple} tied on interface AND
//       shape, four ways: the str/bytes configuration repeated. `list` left for
//       width 9, so it is a THREE-way tie now, and 63 is the number to re-measure
//       after each of the remaining three converts.
//
//       `tuple` has now left for width 14, so on THIS branch it is a TWO-way tie
//       (`set`, `frozenset`) and both of those are converting on another track.
//       **The 63 was NOT re-measured for this conversion, and the reason is a
//       constraint rather than an oversight**: the instrument is a counter on the
//       contract-less `return nullptr` in `common/Ownership.cpp`, and that file
//       was explicitly out of scope for the tuple/int track. Recorded as
//       outstanding rather than quietly skipped -- whoever lands `set` or
//       `frozenset` can take it, and the prediction to check is that the number
//       falls, since the tie is what produced it and the tie is shrinking.
//       Do not read the tuple conversion as having confirmed that.
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
//   before `builtins.str`:  @LyLong_Repr (`builtins.mlir:8614`)
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
//   2. `@LyLong_Repr` still declares no `ly.runtime.result_contract`. It is
//      listed under `str` above, but converting `int` reaches it first from the
//      other side: once `LyLong_DecRef` takes `memref<15xi64>` it stops
//      type-checking against the str header pair, so the single-candidate (C)
//      path stops resolving and falls back to `:450`. That is the benign
//      direction today -- `str` still scores a positive `shapeMatch` and wins --
//      but it means the conversion silently changes which path a str result
//      takes, so declare `result_contract = "builtins.str"` in the same change
//      and do not rely on the fallback.
// ---------------------------------------------------------------------------

} // namespace py::lowering::handle_width
