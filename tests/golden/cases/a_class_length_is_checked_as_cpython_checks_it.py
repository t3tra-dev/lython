# WHAT: what a class's `__len__` returns is a length only after CPython's
# checks (slot_sq_length): a negative answer is a ValueError and one past the
# word an OverflowError, in `len()` and in a truth test alike; an ordinary
# answer counts, and `__bool__` is asked before `__len__`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the checks are on the
# value the method returns at run time. `len(b)` printed -1, `bool(b)`
# answered True, and `len` printed 2**70.
class Box:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class Flag:
    def __init__(self, on: bool) -> None:
        self.on = on

    def __bool__(self) -> bool:
        return self.on

    def __len__(self) -> int:
        return -1


print(len(Box(0)), len(Box(3)), bool(Box(0)), bool(Box(3)))
if Box(2):
    print("nonempty")
if not Box(0):
    print("empty")
print(bool(Flag(False)), bool(Flag(True)))

for n in [-1, 2 ** 70, -(2 ** 70)]:
    b = Box(n)
    try:
        print(len(b))
    except (ValueError, OverflowError) as e:
        print("len:", type(e).__name__, e)
    try:
        print(bool(b))
    except (ValueError, OverflowError) as e:
        print("bool:", type(e).__name__, e)
    try:
        if b:
            print("truthy")
    except (ValueError, OverflowError) as e:
        print("if:", type(e).__name__, e)
