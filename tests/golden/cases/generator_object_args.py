def chars(s: str):
    i = 0
    while i < len(s):
        yield s[i]
        i = i + 1

for c in chars("abc"):
    print(c)

def pairs(xs: list[int], ys: list[str]):
    i = 0
    n = len(xs)
    if len(ys) < n:
        n = len(ys)
    while i < n:
        yield (xs[i], ys[i])
        i = i + 1

for a, b in pairs([1, 2, 3], ["x", "y"]):
    print(a, b)

def wraps(xs: list[str]):
    i = 0
    while i < len(xs):
        yield xs[i]
        i = i + 1

g = wraps(["p", "q", "r"])
print(next(g))
print(next(g))
# abandoned mid-iteration: drop finalizer must release the frame + args
g2 = wraps(["zz"])
print("dropped")

def het(xs: list[int], tag: str):
    i = 0
    while i < len(xs):
        yield (xs[i], tag)
        i = i + 1

for n, t in het([7, 8], "tag"):
    print(n, t)

g3 = het([1], "z")
g3.close()
print("closed")
