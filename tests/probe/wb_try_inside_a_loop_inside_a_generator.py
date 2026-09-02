# A `try` (or a `with`) inside a LOOP inside a generator cannot be compiled:
#
#   unwind cleanup cannot target a handler entry with block arguments
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree). The loop is what makes
# the difference, not the try and not the generator:
#
#   try/finally at the generator's TOP level ....... correct
#   try/except at the generator's top level ........ correct
#   try/finally in a loop in a generator ........... the message above
#   try/except in a loop in a generator ............ the message above
#   try around the whole loop ...................... correct
#   `with` in a loop in a generator ................ the message above
#   a try in a loop in a plain FUNCTION ............ correct
#   the try body without a yield in it ............. the message above
#
# The last line is the useful one: the yield does not have to be inside the
# try. Any `try` inside a generator's loop is refused, which is `for x in xs:`
# with a `try:` in its body -- one of the most ordinary shapes Python has.
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
def g(xs: "list[int]"):
    for x in xs:
        try:
            yield x
        finally:
            pass


print(list(g([1, 2])))
