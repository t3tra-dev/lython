# WHAT: a list, tuple or slice ordering compares the first unequal items with
# the operator the program wrote, as CPython does: `[nan] > [1.0]` is False
# (it is not "unequal and not less"), a class with only `__gt__` orders and
# sorts through the reflected method, a strict subclass's reflected override
# is asked first, the lengths decide when every item is equal, and a refusal
# names the operator and both classes ("'<=' not supported between instances
# of 'int' and 'str'").
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every case here is decided
# by the runtime's boxed comparison over the items' class ids -- the answers,
# which method ran, and the text of the TypeError exist only when it runs.

nan = float("nan")
print([nan] > [1.0], [nan] >= [1.0], (nan,) > (1.0,), [1.0] < [nan], [nan] <= [nan])
class G:
    def __init__(self, v: int) -> None:
        self.v = v
    def __gt__(self, other: "G") -> bool:
        return self.v > other.v
print((G(2),) > (G(1),), [G(1)] < [G(2)])
class L:
    def __init__(self, v: int) -> None:
        self.v = v
    def __lt__(self, other: "L") -> bool:
        return self.v < other.v
print([L(1)] > [L(2)])
try:
    print([L(1)] <= [L(2)])
except TypeError as e:
    print(e)

class A:
    def __init__(self, v: int) -> None:
        self.v = v
    def __lt__(self, other: "A") -> bool:
        print("A.lt")
        return self.v < other.v
class B(A):
    def __gt__(self, other: "A") -> bool:
        print("B.gt")
        return self.v > other.v
print([A(1)] < [B(2)])
print([B(1)] < [A(2)])
print((A(3),) > (B(2),))
class G:
    def __init__(self, v: int) -> None:
        self.v = v
    def __gt__(self, other: "G") -> bool:
        return self.v > other.v
    def __repr__(self) -> str:
        return f"G({self.v})"
print(sorted([G(3), G(1), G(2)]))
class LongNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameName:
    pass
def attempt_more(k: int) -> None:
    try:
        if k == 0:
            print([None] < [1])
        elif k == 1:
            print([1j] >= [2j])
        elif k == 2:
            print([LongNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameName()] > [LongNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameNameName()])
        elif k == 3:
            print((1, 2) <= (1, "x"))
        elif k == 4:
            print([[1, 2], [3]] < [[1, 2], ["a"]])
        elif k == 5:
            print(sorted([3, None, 1]))
        elif k == 6:
            print([b"a"] < [1])
    except TypeError as e:
        print(e)
for k in range(7):
    attempt_more(k)
def four(x: list[object], y: list[object]) -> None:
    print(x < y, x <= y, x > y, x >= y)
four([1, 2], [1, 2, 3]); four([1, 2], [1, 2]); four([2], [1, 9]); four([1.0, 2], [1, 2.5]); four([True], [1]); four([False, 3], [0, 2])
four([b"ab"], [bytearray(b"ac")]); four([bytearray(b"b")], [b"a"]); four([b"x"], [bytearray(b"x")])
nan = float("nan")
print([nan, 1] <= [nan, 2], [1, nan] < [1, nan], (2**60, 1.5) < (2**60 + 1, 0.0))
print(slice(1, 2, 3) < slice(1, 2, 4), slice(1, 3) >= slice(1, 2))

class E:
    pass
ys: list[object] = [1, "a"]
for op2 in range(4):
    try:
        if op2 == 0:
            print(sorted(ys))
        elif op2 == 1:
            print([1] < ["a"])
        elif op2 == 2:
            print((1, E()) <= (1, E()))
        else:
            print([[1]] > [["a"]])
    except TypeError as e:
        print(e)
