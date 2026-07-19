def loud(v: int) -> int:
    print("f", v)
    return v * 2

m = map(loud, [1, 2, 3])
print("created")
print(next(m))
print(next(m))
for v in m:
    print("rest", v)

e = enumerate(["a", "b"])
print(next(e))
i, s = next(e)
print(i, s)
try:
    next(e)
except StopIteration:
    print("done")

z = zip([1, 2, 3], "abc")
a, b = next(z)
print(a, b)

r = reversed([5, 6, 7])
print(next(r))
print(next(r))

it = iter("xy")
print(next(it))
print(next(it))

fl = filter(lambda v: v % 2 == 1, [1, 2, 3, 4, 5])
print(next(fl))
print(next(fl))

e2 = enumerate("ab", 10)
print(next(e2))
it = iter([1, 2])
print(next(it, -1))
print(next(it, -1))
print(next(it, -1))
print(next(it, -1))
s = iter("ab")
print(next(s, "end"))
print(next(s, "end"))
print(next(s, "end"))
