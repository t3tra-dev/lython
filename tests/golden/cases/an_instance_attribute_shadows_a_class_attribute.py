# WHAT: an attribute that is a class attribute and, on some instances, an
# instance attribute too -- stored through another name at module level,
# through `self` in a method other than `__init__`, only on one branch of
# `__init__`, and by a subclass's method -- with a subclass that redeclares
# the class attribute. An instance that stored one reads its own; every
# other instance reads the class's, its RUNTIME class's where a subclass
# redeclares it; the class keeps its own.
#
# WHY THIS IS RUN: which of the two an instance holds is decided by the stores
# its run made. Each store was refused ("would shadow the class attribute"),
# and the conditional store in `__init__` made a read of an instance that did
# not store raise AttributeError, where CPython answers the class attribute.
class C:
    count = 0
    x: int = 0

    def __init__(self, v: int) -> None:
        if v != 0:
            self.x = v

    def bump(self) -> None:
        self.count = self.count + 1


class TestCase:
    maxDiff: int | None = 640
    longMessage: bool = True

    def describe(self) -> str:
        return str(self.maxDiff) + " " + str(self.longMessage)


class Wide(TestCase):
    def setUp(self) -> None:
        self.maxDiff = None


class Quiet(TestCase):
    longMessage = False


c = C(0)
c.count = 5
print(c.count, C.count, C(0).count, c.x, C(3).x, C.x)
d = C(0)
d.bump()
d.bump()
print(d.count, C.count)
w = Wide()
print(w.describe())
w.setUp()
print(w.describe(), TestCase.maxDiff)
q = Quiet()
print(q.describe())
q.longMessage = True
print(q.describe(), Quiet.longMessage)
items: list[TestCase] = [TestCase(), w, q, Quiet()]
print([t.maxDiff for t in items], [t.longMessage for t in items])
