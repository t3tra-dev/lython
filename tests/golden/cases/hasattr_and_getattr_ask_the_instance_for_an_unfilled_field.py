# `hasattr(x, "n")` and `getattr(x, "n", default)` on a field no store may have
# filled yet -- one a method creates, or one the class only annotates -- answer
# from the instance: False and the default before the store, the field after.
# The default form also answers from the class line for a method, a property,
# a class attribute and a missing name, evaluating every argument once.
#
# Why it runs: the same expression has to answer differently before and after
# the store, which only execution shows.
class T:
    def setUp(self) -> None:
        self.n = 3


class Annotated:
    n: int


class P:
    label = "cls-attr"

    def __init__(self) -> None:
        self._v = 2

    @property
    def v(self) -> int:
        return self._v * 10

    def addSkip(self, reason: str) -> None:
        print("skip", reason)


t = T()
print(hasattr(t, "n"), getattr(t, "n", -1))
t.setUp()
print(hasattr(t, "n"), getattr(t, "n", -1))

a = Annotated()
print(hasattr(a, "n"))
a.n = 1
print(hasattr(a, "n"), getattr(a, "n", None))

order: list[str] = []


def mk(tag: str) -> str:
    order.append(tag)
    return tag


p = P()
skip = getattr(p, "addSkip", None)
if skip is not None:
    skip("why")
print(getattr(p, "v", 0), getattr(P, "label", "none"), getattr(p, "nope", "dflt"))
print(getattr(p, "missing", mk("default")), getattr(p, "v", mk("unused")), order)
