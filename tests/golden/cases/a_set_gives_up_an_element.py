# What: `set.pop()` removes AND returns, so the returned element and what is
# left have to agree -- running it is what shows the removal happened once and
# took the element it reported. The empty case raises, which is also only
# visible at run time.
#
# Int sets only: the iteration order of a str set already differs from
# CPython's (its hash is not CPython's), and pop follows the iteration order.
def drain(values: "set[int]") -> "list[int]":
    out: list[int] = []
    while len(values) > 0:
        out.append(values.pop())
    return out


numbers = {1, 2, 3, 4}
first = numbers.pop()
print(first, sorted(numbers), len(numbers))
print(drain({5, 6, 7}))

empty: set[int] = set()
try:
    empty.pop()
except KeyError as error:
    print("empty:", error)
