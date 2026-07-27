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
//      10  _asyncio.Future                                  one-lane
//      12  _asyncio.Task                                    leading word of a
//                                                           multi-input release
//      16  builtins.object                                  payload box / boxed
//                                                           field slot
//      64  types.GeneratorType                              frame
//
// ---------------------------------------------------------------------------
// COLLISIONS, and their measured reachability. Occupancy of single-input
// deallocator interfaces on the merged tree.
//
// IMPORTANT, and a correction to an earlier version of this file: these ties are
// NOT known-live suppressions. Instrumenting the contract-less overload and
// attributing each ambiguous exit BY TYPE (playbook 7.14) found **zero**
// ambiguity on any one-lane width -- 3, 5, 6, 7 or 8 -- across four programs,
// one per contract. All 119 ambiguous exits were on `memref<2xi64>`, i.e. among
// the MULTI-lane contracts (int, str, nullcontext), which is the configuration
// section 8.1 showed to be dangerous and which no conversion here touches.
// One-lane contracts resolved structurally ZERO times, which is the positive
// evidence for the shapeMatch mechanism above: they are never put in a
// structural tiebreak because the contract-named path answers first.
//
// So read the rows below as "a tie whose reachability is unmeasured, and which
// four programs failed to reach", not as a live defect. That makes the case for
// spending a scarce width on one of them weaker, not stronger:
//
//     memref<3xi64>  LyBool, LyFloat, LyReadyIntAwaitable          (3)
//     memref<5xi64>  LyCoroutine, LyRange, LyRangeIterator         (3)
//     memref<2xi64>  LyLong, LyNullContext, LyUnicode              (3)
//                    -- was 4; LyBytes left for width 6 in `00079ef`
//     memref<4xi64>  LyAsyncCounter, LyCounter, LyReadyAsyncCounter (3)
//     memref<8xi64>  LyBytesIO, LyDict, LyEventLoop, LyFileIO,
//                    LyFunction, LyStringIO, LyTextIO              (7)
//     memref<6xi64>  LyBytes                                       (1)  clean
//     memref<7xi64>  LyComplex                                     (1)  clean
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
inline constexpr int kFreeHandleWidths[] = {9, 11, 13, 14, 15};

// The scarcity a converter should see before assuming a width is available,
// and -- more importantly -- the ties that are ALREADY suppressing groups by
// mechanism (A) today, before any further conversion:
//
//   (memref<2xi64>, memref<2xi64>, memref<?xi64>)  x63 declarations
//       builtins.list, set, frozenset, tuple. Tied on interface AND shape, four
//       ways: the str/bytes configuration repeated. PREDICTION, unmeasured:
//       these four groups are suppressed today, so converting the first of them
//       should surface whatever the tie has been hiding -- the same way breaking
//       the str/bytes tie surfaced the exception family's double free. Budget for
//       that rather than treating it as a regression from the conversion.
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
//   before `builtins.int`:  @__ly_boxed_long_view (`builtins.mlir:15405`)
//       Reads box pointer words 5 and 6, i.e. int's LANE 1 and LANE 2 addresses,
//       which one-laning leaves uninitialised. The last lane-indexed raw box
//       accessor in the tree. Three callers, so equality, ordering and float
//       coercion all route through it. Invisible to every static oracle -- see
//       the playbook's blocker note for the full statement.
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

} // namespace py::lowering::handle_width
