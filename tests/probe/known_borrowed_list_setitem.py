# probe: REPORTED passes: list setitem on a borrowed (parameter) list (in place)
# axes: acquire=param width=w3list op=setitem flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 42

def put(xs: list[int]) -> None:
    xs[0] = 42


xs: list[int] = [1]
put(xs)
print(xs[0])
