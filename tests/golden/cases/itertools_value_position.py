from itertools import count, chain, islice, pairwise, dropwhile, repeat, cycle

c = count(5, 2)
print(next(c), next(c), next(c))
g = chain([1, 2], [9])
print(next(g))
for v in g:
    print(v)
s = islice("abcdef", 1, 5, 2)
print(next(s))
for x in s:
    print(x)
p = pairwise([3, 5, 7])
a, b = next(p)
print(a, b)
d = dropwhile(lambda x: x < 4, [1, 5, 2])
print(next(d), next(d))
r = repeat("q", 2)
print(next(r), next(r))
cy = cycle([1, 2])
print(next(cy), next(cy), next(cy))
