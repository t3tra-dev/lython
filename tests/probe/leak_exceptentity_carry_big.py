# probe: leak -- an except-bound exception entity carried out of the statement
#   into a local that outlives it (40000 iterations)
# axes: op=leak-loop acquire=except width=wNexc iterations=40000
# CLASSIFICATION @ 1440121: 1 正しい (49.7 B/iteration
#   against a 500 B floor)
# CPython 3.14 expects: 160000
#
# Same program as leak_exceptentity_carry_small with the iteration count
# changed: the difference in peak RSS over the difference in iterations is what
# the loop failed to release.
def once() -> int:
    kept: BaseException = ValueError("init")
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = e
    return len(str(kept))


total = 0
for _ in range(40000):
    total += once()
print(total)
