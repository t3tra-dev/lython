# What: the name a loop, a try or a with leaves behind, computed from another
# name the same body binds. Its slot has to be typed before the body runs, so
# everything the value depends on has to be in scope by then -- including a
# container the body fills after it creates it. Each result is decoded (added
# to, indexed, compared) so an erased slot would not survive the read.
def from_a_local() -> int:
    for _ in range(1):
        base = 5
        doubled = base * 2
        total = doubled + 1
    return total


def from_a_call() -> int:
    def scale(n: int) -> int:
        return n * 3

    for _ in range(1):
        seen = scale(4)
        total = seen + 1
    return total


def from_a_filled_list() -> int:
    for _ in range(1):
        xs = []
        xs.append(7)
        total = xs[0] + 1
    return total


def from_a_filled_dict() -> int:
    try:
        counts = {}
        counts["k"] = 9
        total = counts["k"] + 1
    finally:
        pass
    return total


def from_an_extended_list() -> int:
    for _ in range(1):
        xs = []
        xs.extend([11, 12])
        total = xs[1] + 1
    return total


def from_an_augmented_list() -> int:
    for _ in range(1):
        xs = []
        xs += [13]
        total = xs[0] + 1
    return total


class Resource:
    def __enter__(self) -> int:
        return 1

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


def from_a_with_body() -> str:
    with Resource():
        word = "ab"
        label = word * 2
    return label


print(from_a_local(), from_a_call(), from_a_filled_list())
print(from_a_filled_dict(), from_an_extended_list(), from_an_augmented_list())
print(from_a_with_body(), len(from_a_with_body()))


# The name that carries its own previous value across the back edge still does.
def accumulate() -> int:
    running = 0
    for i in range(4):
        running = running + i
    return running


print(accumulate())
