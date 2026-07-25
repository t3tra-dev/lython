# probe: borrowed list pop (shrink, no realloc)
# axes: acquire=param width=w3list op=pop flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1

def drop(xs: list[int]) -> None:
    xs.pop()


xs: list[int] = [1, 2]
drop(xs)
print(len(xs))
