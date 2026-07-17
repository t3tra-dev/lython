# Cross-check (p2-genframe x p2-traceback): a generator suspended inside an
# except handler parks its in-flight exception WITH its __context__ chain;
# unrelated exceptions raised and handled between resumptions neither adopt
# the parked chain nor resurrect it, and the body's later raise still chains
# to the parked handler exception.
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
try:
    next(g)
except RuntimeError as e:
    print("caught RuntimeError:", e)
print("done")
