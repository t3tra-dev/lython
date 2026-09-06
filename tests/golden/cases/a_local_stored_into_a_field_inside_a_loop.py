# What: a value created OUTSIDE a loop stored into a field INSIDE it. The store
# may only MOVE a token whose whole life it can see, and the walk that decides
# that asked "does anyone else need this value" -- for
#
#     w = "abc"
#     for _ in range(1):
#         h.tag = w
#
# nobody does, so the store took the token and the next trip ran the same store
# with the same one: "released or transferred more than once on one CFG path",
# out of the LOWERING, for a program that stores a local into a field in a loop.
# The store's own REPETITION is what outlives it -- one definition, N moves.
#
# Running it is what shows the reference survived: the field is read after the
# loop and after the object outlives the trip that wrote it. The nested loop is
# here because a definition in an OUTER loop and a store in an INNER one sit in
# one strongly-connected component and the store still repeats per definition;
# the loop-target store beside it is the shape that always worked and must keep
# moving rather than retaining.
class Holder:
    def __init__(self) -> None:
        self.tag: str = ""
        self.n: int = 0


def outer_local() -> str:
    w = "abc"
    h = Holder()
    for _ in range(3):
        h.tag = w
    return h.tag


def nested(rounds: int) -> str:
    h = Holder()
    for _ in range(rounds):
        w = "x" * 3
        for _ in range(3):
            h.tag = w
    return h.tag


def loop_target(items: list[str]) -> str:
    h = Holder()
    for s in items:
        h.tag = s
    return h.tag


def through_a_local_alias() -> int:
    w = "abcd"
    for _ in range(2):
        v = w
    return len(v)


def while_form() -> str:
    w = "zz"
    h = Holder()
    i = 0
    while i < 2:
        h.tag = w
        i += 1
    return h.tag


print("outer local", outer_local())
print("nested", nested(2))
print("loop target", loop_target(["a", "bb"]))
print("local alias", through_a_local_alias())
print("while", while_form())
