# probe: the except-bound exception ENTITY assigned to a loop-carried binding
# axes: acquire=except width=wNexc op=rebind flow=loop
# CLASSIFICATION @ kernel/lane-dict 7cd3b94: 2 silent 誤実行 (prints the stale
#   pre-loop value)
# CLASSIFICATION @ main 1c3dfc4: 3 loud 拒否 -- but from the ownership verifier
#   ("released owned resource from @LyValueError_New is used after release"),
#   which --release turns off: measured there as a SIGSEGV in the JIT. So the
#   "also reproduces at main 1c3dfc4" line above is wrong for this probe as
#   recorded; the stale print is what the five-lane dict binary did.
# CLASSIFICATION @ 1440121: 3 loud 拒否 (診断) -- rejected in the emitter, the
#   same answer with and without --release. Neither channel can carry the entity
#   here: a lane publishes the borrowed pointer past the handler's discard, and
#   the storage promotion is withheld from loop-carried locals because moving a
#   loop block argument's token into an aggregate slot double-frees.
# CPython 3.14 expects: boom
#
# Written to decide whether the exception family's transfer_args reaches a
# loop-carried group, which is the condition all three known owner-group defects
# needed. It does not get that far: the assignment is already lost without a loop
# (rebind_exceptentity_straight), so the loop is not the discriminator here.
kept: BaseException = ValueError("init")
i = 0
while i < 3:
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = e
    i += 1
print(str(kept))
