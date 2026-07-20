# sorted(<non-list iterable>) materializes through the list constructor,
# and d.keys() in value position iterates the dict itself (the keys view
# is a phantom with no runtime representation).


def gen():
    yield 3
    yield 1
    yield 2


print(sorted(gen()))
print(sorted("cba"))
print(sorted((5, 4, 6)))
print(sorted({3, 1, 2}))
print(sorted([2, 1]))
print(sorted([3, 1, 2], reverse=True))
print(sorted(["bb", "a"], key=lambda w: len(w)))
d = {"b": 2, "a": 1, "c": 3}
print(sorted(d.keys()))
print(sorted(d))
print(list(d.keys()))
print(sorted(d.keys(), reverse=True))
ks = [k for k in d.keys()]
print(ks)
print(sorted(set(d.keys())))
