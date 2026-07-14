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
