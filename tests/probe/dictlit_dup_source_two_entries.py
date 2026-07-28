# ✅ REPAIRED 2026-07-28. A SECOND, INDEPENDENT DEFECT found while measuring the
# frequency one, and the reason this is a separate file: it needs NO loop, and
# the frequency query does not touch it. Ablating them one at a time separates
# them (`LYTHON_ABLATE_DICT_SOURCE_MOVE_DEDUP=1` reproduces only this).
#
# A source hands over the ONE token it holds, so a source filling several slots
# may be released ONCE. `initializeSequencePayload` has always deduped on the
# element's physical identity -- its comment cites `(j, j)`, two slots of ONE
# literal. `initializeDictPayload` had no dedup at all, and the dict spelling
# repeats across ENTRIES rather than within one, so the sequence repair never
# covered it.
#
# ⚠️ THIS IS WHY "SAME CODE, SAME DEFECT" WAS THE WRONG STARTING ASSUMPTION IN
# BOTH DIRECTIONS. The dict path was missing something the sequence path had, on
# top of sharing the gap the sequence path had just closed.
#
# Measured on bcfbbf9, 5 reps each:
#   str source, value read back    -> WWWWW  (silent: prints 0, not 3)
#   heap-int source, len() only    -> XX...  (intermittent abort)
# The str spelling is the one that matters: it never aborts, so an exit-code-only
# check calls it clean 5 times out of 5.
#
# ⚠️ The binding must have NO use outside the literal, or `valueIsConsumedOnlyBy`
# is already false and no move is attempted. `x = "q" + "rs"` then two entries and
# nothing else -- adding a `print(x)` makes the whole shape unreachable, which is
# how a first attempt at this spelling came out clean and was nearly recorded as
# "does not reproduce".
sx = 0
for i in range(1):
    x = "q" + "rs"
    dup = {"a": x, "b": x}
    sx += len(dup["a"])
print(sx)  # CPython 3.14: 3

ix = 0
for i in range(1):
    y = i + 345
    dupi = {"a": y, "b": y}
    ix += len(dupi)
print(ix)  # CPython 3.14: 2
