# Cross-check (p2-genframe x p2-traceback): the chained exception parked by
# a handler-suspended generator must not leak into the process chain state.
# The uncaught IndexError reports WITHOUT any "During handling of ..."
# section (the suspended generator's ValueError/TypeError belong to its
# frame, not to this raise).
from collections.abc import Generator


def gen() -> Generator[int, None, None]:
    try:
        raise ValueError("first")
    except ValueError:
        try:
            raise TypeError("second")
        except TypeError:
            yield 1
            raise RuntimeError("third")


g = gen()
print(next(g))
try:
    raise KeyError("unrelated")
except KeyError:
    print("handled unrelated")
raise IndexError("final")
