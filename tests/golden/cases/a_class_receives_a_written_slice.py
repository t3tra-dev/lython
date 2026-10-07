# WHAT: `obj[a:b:c]` on a class that defines the dunder calls it with the
# slice object BUILD_SLICE makes -- slice(a, b, c), None for each absent part
# -- for __getitem__, __setitem__ and __delitem__ alike; a key declared
# `int | slice` narrows with isinstance and reads the slice through
# indices(); and the subscript makes a real slice even where a local named
# `slice` hides the builtin.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the dunder bodies print
# the slice they were handed and act on it, so the bounds the object carries
# and the order the bodies run in are run-time output.


class Seq:
    def __init__(self, items: list[int]) -> None:
        self.items = items

    def __getitem__(self, key: slice) -> list[int]:
        print("get", key)
        return self.items[key]

    def __setitem__(self, key: slice, value: list[int]) -> None:
        print("set", key, value)
        self.items[key] = value

    def __delitem__(self, key: slice) -> None:
        print("del", key)
        del self.items[key]


class Either:
    def __init__(self, items: list[int]) -> None:
        self.items = items

    def __getitem__(self, key: int | slice) -> int | list[int]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self.items))
            return [self.items[i] for i in range(start, stop, step)]
        return self.items[key]


def shadowed(q: Seq) -> list[int]:
    slice = 3
    return q[slice:]


q = Seq([0, 1, 2, 3, 4, 5])
print(q[1:3], q[::2], q[:], q[4:], q[:-1], q[1:5:2])
k = slice(2, 4)
print(q[k])
q[0:2] = [7, 7, 7]
print(q.items)
del q[::3]
print(q.items)
print(shadowed(Seq([0, 1, 2, 3, 4, 5])))
e = Either([10, 11, 12, 13, 14])
print(e[1], e[1:4], e[::-2], e[-1])
