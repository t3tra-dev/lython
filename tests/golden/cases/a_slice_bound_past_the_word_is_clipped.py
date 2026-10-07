# WHAT: a slice bound or a find window past the 64-bit word is the nearest end
# of the word, as CPython reads an index (_PyEval_SliceIndex, which is
# PyNumber_AsSsize_t(v, NULL)) -- `xs[:2**70]` is the whole list, not an
# OverflowError: list/str/tuple/bytes/range slices, slice assignment and
# deletion, a step past the word either way, str and bytes find/rfind/index/
# count/startswith/endswith windows, and a bound that grew past the word in a
# loop (a deferred int, whose word is not its value).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the bound is an int object
# read at run time; whether it clips or raises, and what the clipped slice
# holds, is only seen by running it.
xs = [1, 2, 3]
big = 2**70
print(xs[:2**70], xs[-2**70:], xs[::2**70], xs[::-2**70], xs[2**70:], xs[:-big])
print("abc"[:big], (1, 2, 3)[-big:], b"abc"[::-big], range(10)[-big:big:3])
ys = [1, 2, 3, 4]
ys[big:] = [9]
print(ys)
del ys[:-big]
print(ys)
del ys[-big:big]
print(ys)
print("abcabc".find("c", -big, big), "abcabc".rfind("a", 0, big), "abcabc".count("b", -big))
print("abc".startswith("b", 1, big), "abc".endswith("b", -big, -1), "abcabc".index("c", 3, big))
print(b"abcabc".find(b"c", 3, big), b"abcabc".count(b"a", -big, big), b"abc".endswith(b"c", 0, big))
n = 3
for k in range(66, 72):
    n = n * 2
print(xs[:n], xs[-n:], xs[::-n], "abc".find("c", 0, n))
