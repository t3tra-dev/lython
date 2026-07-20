// Contract manifest for stdlib `itertools`.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/itertools.pyi
//
// This manifest declares the module surface only. Every itertools call
// compiles in the emitter (EmitterIterators.cpp) to a loop fusion or to a
// per-call-site synthesized generator function over indexable sequences -
// the same machinery the enumerate/zip/map/filter builtins use - so no
// native function bodies live here. Calls the desugars cannot express are
// rejected with a diagnostic at emit time; nothing reaches runtime lowering.
//
// Deviations from CPython (pending language surface, enforced loudly unless
// noted):
//   - value-position combinators require indexable sequence sources
//     (list/str/tuple/bytes); iterator/generator sources work in for-loop
//     position through fusion
//   - cycle() re-reads its sequence instead of building the saved-element
//     list, so mutating the sequence mid-iteration is observable (CPython
//     replays the first pass's snapshot)
//   - product/combinations read their pools lazily instead of snapshotting
//     them at call time (observable only by mutating a pool between the
//     call and the iteration)
//   - islice stop/start/step accept negative values only as compile-time
//     constants, where they are rejected statically; runtime values are
//     range-checked where a guard is possible
//   - fused accumulate (and pairwise over a generator source) pre-seed the
//     loop-carried slot with an int, restricting those forms to int
//     elements until the ownership planner supports branch-local first
//     assignments; other element types fail the type merge loudly
//   - accumulate(initial=), takewhile/filterfalse/accumulate/zip_longest/
//     chain.from_iterable as first-class values, and islice over a NAMED
//     iterator (the driving loop would over-consume it) are diagnosed at
//     emit time with a rewrite hint
//   - groupby / tee / batched / starmap / permutations are not implemented
//     yet (diagnosed at emit time)

module attributes {
  ly.typing.module = "itertools",
  ly.typing.callable_exports = [
    "itertools.accumulate",
    "itertools.batched",
    "itertools.chain",
    "itertools.combinations",
    "itertools.combinations_with_replacement",
    "itertools.count",
    "itertools.cycle",
    "itertools.dropwhile",
    "itertools.filterfalse",
    "itertools.groupby",
    "itertools.islice",
    "itertools.pairwise",
    "itertools.permutations",
    "itertools.product",
    "itertools.repeat",
    "itertools.starmap",
    "itertools.takewhile",
    "itertools.tee",
    "itertools.zip_longest"
  ]
} {
}
