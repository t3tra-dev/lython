# WHAT: an `object` local bound to an int and rebound to an instance in one
#   arm of an `if` is the instance after the join: repr calls its __repr__,
#   a dict keys on it by its __eq__/__hash__, and isinstance sees it.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was the value
#   handed across the join -- the instance's header taken for a box -- which
#   compiles either way; only the run shows the crash or the wrong key.
class P:
    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, P) and other.x == self.x

    def __hash__(self) -> int:
        return self.x

    def __repr__(self) -> str:
        return "P" + str(self.x)


def key(i: int) -> object:
    k: object = i
    if i % 2 == 1:
        k = P(i)
    return k


def show(i: int) -> None:
    k: object = P(0)
    if i > 0:
        k = P(i)
    print(repr(k), isinstance(k, P))


table: dict[object, int] = {}
for i in range(6):
    k: object = i
    if i % 2 == 1:
        k = P(i)
    table[k] = i
print(table)
print(len(table), sum(1 for i in range(6) if key(i) in table))
show(0)
show(3)
