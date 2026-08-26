# probe: leak -- a `list[int] | None` local carried across a `for` loop's back edge (20000 iterations)
# axes: op=leak-loop iterations=20000
#
# `acquireUnionCarriedTokens` is called from emitWhile, emitFor AND
# emitAsyncFor; `releaseUnionCarriedTokens` from emitWhile alone. This weighs
# what the two loop shapes without the release cost. Until the union release
# stopped being tag-guarded, a union carried by a `for` was refused before the
# leak could be weighed at all.
#
# The carried member is a LIST and not a str: a leaked str is about 41 B per
# iteration, an order of magnitude under this instrument's 500 B floor, so the
# same shape spelled with a str reports "no leak" whether or not one is there.
#
# CPython 3.14 expects: 120000

def once() -> int:
    s: list[int] | None = [1, 2, 3, 4, 5, 6, 7, 8]
    n = 0
    for x in [1, 2, 3]:
        n += x
        if x == 2:
            s = [9, 8, 7, 6, 5, 4, 3, 2]
    if s is None:
        n += 100
    return n


total = 0
for _ in range(20000):
    total += once()
print(total)
