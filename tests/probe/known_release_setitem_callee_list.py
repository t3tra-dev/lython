# probe: REPORTED loud (B8): setitem on a list returned by a callee
# axes: acquire=call width=w3list op=setitem flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 9 3

def make() -> list[int]:
    return [1, 2, 3]


xs = make()
xs[0] = 9
print(xs[0], len(xs))
