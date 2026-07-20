# One boxed value filling several literal slots must not double-consume its
# source token when the container dies (`print(t)` releases the tuple).


def pick(n: int) -> int:
    return n + 10


i = pick(1)
j = pick(2)
t = (1, i, j, j)
print(t)
print(t[3], j)
xs = [j, j, i, j]
print(xs, j, i)
print(len(xs), t[1] + xs[0])
