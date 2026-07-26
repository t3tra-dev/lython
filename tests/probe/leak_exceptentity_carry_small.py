# probe: leak -- an except-bound exception entity carried out of the statement
#   into a local that outlives it (100 iterations)
# axes: op=leak-loop acquire=except width=wNexc iterations=100
# CLASSIFICATION @ 1440121: 1 正しい
# CPython 3.14 expects: 400
#
# The carry goes through a storage cell allocated per execution of the
# statement, so a missed release of either the cell or the entity it retains
# shows up here rather than in the correctness probes.
def once() -> int:
    kept: BaseException = ValueError("init")
    try:
        raise ValueError("boom")
    except ValueError as e:
        kept = e
    return len(str(kept))


total = 0
for _ in range(100):
    total += once()
print(total)
