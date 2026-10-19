# A class object is true -- its truth is its metaclass's, and every class
# here has `type`'s. `if t:` over `type[int] | None` was refused: the
# evidence asked the INSTANCE contract and found `int.__bool__`, a method of
# ints. `sys.exc_info()[0]` in a condition reached the lowering and failed
# there for the same reason.
# Why run it: the answer is a value, True for the class and False for None.
import sys


def pick(flag: bool) -> type[int] | None:
    return int if flag else None


t = pick(True)
if t:
    print("class", t.__name__)
print(bool(pick(False)), bool(pick(True)), not pick(True))
try:
    raise KeyError("k")
except Exception:
    kind = sys.exc_info()[0]
    print(kind.__name__ if kind else None)
print("nothing handled" if not sys.exc_info()[0] else "handled")
