# probe: a value DERIVED from the except-bound exception assigned to a
#   loop-carried binding -- the control for rebind_exceptentity_loop
# axes: acquire=except width=w1str op=rebind flow=loop
# CLASSIFICATION @ kernel/lane-dict 7cd3b94: 1 正しい
# CPython 3.14 expects: boom
#
# Same binding, same position, same loop as rebind_exceptentity_loop, differing
# only in whether what escapes is the exception or a str derived from it. This is
# what isolates the defect to the ENTITY escaping: it is not the loop, and it is
# not general binding visibility out of an except block.
kept: str = "init"
i = 0
while i < 3:
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = str(e)
    i += 1
print(kept)
