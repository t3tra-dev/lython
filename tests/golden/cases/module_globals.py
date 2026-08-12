# Storage-backed module globals: annotated module-level assignments of the
# immutable scalars and user classes park one retained reference in
# process-lifetime cells; functions read them, user-class state mutates in
# place across calls, and rebinding releases the previous holder.
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


NAME: str = "first"
ORIGIN: Point = Point(1, 2)
RATIO: float = 1.5
FLAG: bool = True
DATA: bytes = b"seed"
LIMIT: int = 7


def describe() -> str:
    return NAME


def origin_sum() -> int:
    return ORIGIN.x + ORIGIN.y


def scale() -> float:
    return RATIO


def enabled() -> bool:
    return FLAG


def payload() -> bytes:
    return DATA


def limit() -> int:
    return LIMIT


def rebind() -> None:
    global NAME, ORIGIN
    NAME = "second"
    ORIGIN = Point(10, 20)


print(describe())
print(origin_sum())
print(scale())
print(enabled())
print(payload())
print(limit())
rebind()
print(describe())
print(origin_sum())


# Why execution: a module-scope name bound ONCE to a literal is not a cell --
# it re-emits the literal at every reference -- so only running it shows the
# function reads the same value the module does. `N = 5` was "unresolved name
# 'N'" from inside a function while `N: int = 5` worked; CPython does not
# distinguish the two spellings.
UNANNOTATED_INT = 42
UNANNOTATED_STR = "plain"
UNANNOTATED_FLOAT = 0.25
UNANNOTATED_BOOL = False


def read_literals() -> str:
    return (
        str(UNANNOTATED_INT)
        + " "
        + UNANNOTATED_STR
        + " "
        + str(UNANNOTATED_FLOAT)
        + " "
        + str(UNANNOTATED_BOOL)
    )


print(read_literals())
print(UNANNOTATED_INT, UNANNOTATED_STR, UNANNOTATED_FLOAT, UNANNOTATED_BOOL)
