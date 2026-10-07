# WHAT: a slice OBJECT indexes a sequence the way a written slice does --
# `xs[s]` on a list, tuple, str, bytes and range, `xs[s] = ys` and `del xs[s]`
# on a list (extended slices included), bounds past the word clipped and a
# zero step refused -- while a dict takes the same object as a key it hashes,
# assigns and deletes.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the slice's parts are read
# out of the object at run time, so what is selected, spliced or removed is
# only visible in the values the program prints afterwards.


def firsts(n: int) -> slice:
    return slice(n)


def big(n: int) -> int:
    return 2**n


xs = [0, 1, 2, 3, 4, 5]
s = slice(1, 5, 2)
print(xs[s], (0, 1, 2, 3, 4, 5)[s], "abcdef"[s], b"abcdef"[s], range(10)[s])
r = slice(None, None, -1)
print(xs[r], "abc"[r], range(5)[r], (1, 2, 3)[r], b"xyz"[r])
print(xs[firsts(3)], "hello"[firsts(2)], xs[slice(big(70), None)], xs[slice(-big(70), 2)])
ys = [0, 1, 2, 3, 4]
ys[slice(1, 3)] = [9, 9, 9]
print(ys)
del ys[slice(None, None, 2)]
print(ys)
zs = [0, 1, 2, 3, 4, 5]
zs[slice(None, None, 2)] = [7, 8, 9]
del zs[firsts(1)]
print(zs)
try:
    print(xs[slice(None, None, 0)])
except ValueError as e:
    print("ValueError", e)
d = {slice(1, 2): 1}
d[slice(1, 2)] = 3
d[slice(None)] = 4
print(d[slice(1, 2)], len(d))
del d[slice(1, 2)]
print(d)
