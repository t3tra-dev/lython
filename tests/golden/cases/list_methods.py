# sort (stability across the numeric tower), reverse, copy, concat,
# repeat, comparisons, count/index/remove, membership.
xs = [3, 1, 2, 1]
xs.sort()
print(xs)
xs.reverse()
print(xs)
print(xs.copy(), xs.copy() == xs)
print(xs + [9, 8], [0] * 4, [1, 2] * 0)
print([1, 2] == [1, 2], [1, 2] < [1, 3], [1, 2] < [1, 2, 0], [2] >= [1, 9])
print(xs.count(1), xs.index(2))
xs.remove(1)
print(xs, 1 in xs, 5 in xs)
# Stable sort keeps the order of equal elements: 0 == False, 1 == 1.0 == True.
mixed = [1.0, True, 0, False, 1]
mixed.sort()
print(mixed)
words = ["pear", "fig", "apple", "fig"]
words.sort()
print(words)
print(sorted([5, -1, 3]), sorted(["b", "a"]))
base = [4, 2]
print(sorted(base), base)
fl = [2.5, -0.5, 2.25]
fl.sort()
print(fl)
ts = [(2, 1), (1, 9), (1, 2)]
ts.sort()
print(ts)
print((1, 2) in ts, (9, 9) in ts)
