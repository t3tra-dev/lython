# probe: the except-bound exception ENTITY assigned to a loop-carried binding
# axes: acquire=except width=wNexc op=rebind flow=loop
# CLASSIFICATION @ kernel/lane-dict 7cd3b94: 2 silent 誤実行 (prints the stale
#   pre-loop value; also reproduces at main 1c3dfc4)
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
