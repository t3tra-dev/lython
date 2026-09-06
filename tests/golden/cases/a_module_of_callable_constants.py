# Helper for a_callable_global_is_read_from_a_function. A module-level lambda
# is the shape a library uses for a small policy hook, and it had no symbol to
# bind and no literal spelling, so it resolved from nowhere across the boundary.
from typing import Callable

FACTOR = 3
SCALE: Callable[[int], int] = lambda n: n * FACTOR


def apply(n: int) -> int:
    return SCALE(n)
