# A `try` (or a `with`) inside a LOOP inside a generator cannot be compiled:
#
#   unwind cleanup cannot target a handler entry with block arguments
#
# RE-MEASURED (2026-09-07, RelWithDebInfo). ⭐ THE LOOP IS NOT THE LINE. It is
# for a `try`; for a `with` there is no line at all -- every `with` inside a
# generator is refused, including the flattest one there is:
#
#   try/finally at the generator's TOP level ....... correct
#   try/except at the generator's top level ........ correct
#   try around the whole loop ...................... correct
#   a try in a loop in a plain FUNCTION ............ correct
#   a `with` in a plain FUNCTION, loop and all ..... correct
#   try/finally in a loop in a generator ........... the message above
#   try/except in a loop in a generator ............ the message above
#   the try body without a yield in it ............. the message above
#   `with X() as n:` then `yield n` ................ the message above
#   `with X():` then `yield 1` ..................... the message above
#   `with X() as n:` with NO yield inside it ....... the message above
#   a `with` AFTER a yield ......................... the message above
#   a `with` in a loop, or a loop in a `with` ...... the message above
#
# So the refused set is: any `try` inside a generator LOOP, and any `with`
# inside a generator at all. `with open(p) as f: for line in f: yield line` is
# the idiom this costs, and `for x in xs:` with a `try:` in its body is the
# other -- two of the most ordinary shapes Python has.
#
# ⭐ WHERE IT COMES FROM: a generator's loop is flattened into a resume state
# machine, so its blocks carry the frame's live lanes as BLOCK ARGUMENTS -- the
# handler entry in the failing program takes six (i64, i1) pairs where the
# non-generator spelling of the same loop takes none. The unwind cleanup that
# releases held tokens ends with `cf.br handler`, and a branch to a block with
# arguments needs operands.
#
# ⛔ AND THE OPERANDS ARE NOT AVAILABLE WHERE THE BRANCH IS: the cleanup block
# hangs off an anchor `cond_br` wired into the MIDDLE of the block holding the
# call, and the values the handler's normal predecessors pass are computed in
# the tail that the anchor splits off -- so they do not dominate the cleanup.
# Recovering them means knowing which SSA value stands for each handler
# argument at the throwing point, which is a question the cleanup placement
# does not ask today and cannot answer from what it holds
# (`getOrCreateCleanupHandler`, Runtime/Passes/Ownership.cpp).
#
# ⭐ RE-MEASURED 2026-09-05, and the `with` row above needed a fix of its own to
# get here: the yield-type walk did not bind a `with ... as X` target, so
# `with Ctx() as base: yield base` was refused EARLIER, with "annotated
# Iterator[int] but yields builtins.object" -- a sentence about an annotation
# that was correct. It now reaches this limit like the others.
#
# ⛔ The note that a target-less `with` reaches a THIRD limit ("generator
# resume continuation live closure violated") is STALE as of 2026-09-07: it
# reaches this one. Nothing about the `with` spelling changes the answer any
# more.
#
# ⛔ TWO OPERAND RULES TRIED AND DROPPED, 2026-09-06. The cleanup block CAN be
# given the handler's block arguments (`getOrCreateCleanupHandler` takes them,
# memoised alongside the group set so two sites cannot share one arm while
# passing different values) -- the question is only WHICH values, and both
# answers produce IR that LLVM rejects with "Instruction does not dominate all
# uses" on the handler's phi:
#
#   - the operands the handler's normal predecessor passes, required to
#     dominate the ANCHOR: a value computed in the try BODY has not run on
#     every path that reaches the landing pad the final EH phase makes.
#   - the same, required to dominate the HANDLER BLOCK: still rejected, and
#     the phi it breaks names the ORIGINAL predecessor's incoming value --
#     which says the split itself (`head->splitBlock(anchorBefore)`) moves a
#     definition out from under a use, not just the new edge.
#
# So the next attempt has to reason about the CFG the EH phase produces, not
# the one the cleanup placement sees. Neither rule is a matter of picking a
# better dominance query.
#
# ⭐ WHAT THE FAILING IR ACTUALLY LOOKS LIKE, read 2026-09-07 on the FLATTEST
# case (`with G() as n: yield n`, no loop): the handler entry takes ONE
# argument, `memref<5xi64>` -- the context manager object -- and the block that
# branches to it passes its OWN block argument, not a definition. The same
# object has a different SSA name in every block the flattening threaded it
# through. So the missing input is a map from `handler argument index` to `the
# name that value has AT THE ANCHOR`, which is a forward dataflow walk from the
# anchor to the handler through the NORMAL edges -- not a dominance query at
# all, and not something either dropped rule could have gotten right.
#
# ⛔ And a sibling shape that does NOT need a try at all is recorded separately
# in wb_a_short_circuit_guard_around_a_yield.py: a short-circuit `and`/`or`
# guarding a yield loses a list local's unwind release, where the same condition
# written as nested `if`s compiles. Both are the refcount phase failing on an
# exceptional edge inside a generator, and both work outside one.
def g(xs: "list[int]"):
    for x in xs:
        try:
            yield x
        finally:
            pass


print(list(g([1, 2])))
