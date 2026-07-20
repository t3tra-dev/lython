xs = [1, 2, 3]
ys = list(xs)
ys.append(4)
print(xs, ys)
t = tuple(xs)
print(t, len(t))
s = set([3, 1, 2, 3])
print(sorted(s))
e = list()
d = dict()
tt = tuple()
print(e, d, tt)
d2 = dict(a=1, b=2)
print(d2)
src = {"x": 10, "y": 20}
cp = dict(src)
cp["z"] = 30
print(src, cp)
ls = list("abc")
print(ls)
lr = list(range(4))
print(lr)
sq = list(x * x for x in [1, 2, 3])
print(sq)
st = set(x % 3 for x in [3, 4, 5, 6])
print(sorted(st))
tp = tuple(x + 1 for x in [1, 2])
print(tp)
pairs = [(1, "a"), (2, "b")]
dp = dict(pairs)
print(dp)
t2 = tuple((1, 2, 3))
print(t2)
xs: list[int] = list()
xs.append(1)
print(xs)
d: dict[str, int] = dict()
d["k"] = 5
print(d)
s: set[int] = set()
print(len(s))
ss = set("aab")
print(sorted(ss))
ts = tuple([10, 20]) 
print(ts[0] + ts[1])
def gen():
    yield 1
    yield 5
lg = list(gen())
print(lg)
sg = set(gen())
print(sorted(sg))
tg = tuple(gen())
print(tg)
nested = list([[1], [2]])
print(nested)
c = list(xs)
c.append(9)
print(xs, c)
def dup(xs: list[int]) -> list[int]:
    return list(xs)

def freeze(xs: list[int]) -> tuple[int, ...]:
    return tuple(xs)

def use(ys: list[int]) -> int:
    return len(ys)

base = [1, 2, 3]
print(dup(base), freeze(base))
print(use(list(base)))
r = dup(list(base))
print(r)
for v in list(base):
    print(v)
