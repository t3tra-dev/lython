# WHAT: `result = f(arg)` with `f: Callable[[str], object]` inside a `try`
# whose handlers all raise, read after the `try` -- argparse's `_get_value`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which handler ran, if
# any, and so what reaches the read, depends on the argument.
#
# ⛔ An `object` value was not one the `try` carried out of its body, so the
# read was "unresolved name 'result'".
from collections.abc import Callable


def get(f: Callable[[str], object], arg: str) -> object:
    try:
        result = f(arg)
    except ValueError:
        raise KeyError(arg)
    return result


print(get(int, "3"), get(str.upper, "x"))
try:
    get(int, "zz")
except KeyError as e:
    print("KeyError", e)
