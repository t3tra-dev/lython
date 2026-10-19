# WHAT: a test framework's base in another module declares class attributes
# -- a class value, a bool, an `int | None` -- and reads them in its own
# method. Test classes here redeclare one at class level, store others on
# the instance in setUp (the class value included), or both, and a store
# through a test instance at module level shadows one too. The base's method
# reads what each instance has: its own, or its runtime class's.
#
# WHY THIS IS RUN: what the base's method reads is decided by the instance's
# class and its stores at run time. `self.maxDiff = None` in a setUp -- a
# unittest fixture's commonest line -- was refused.
from an_imported_class_attribute_shadowed_by_an_instance_lib import TestCase


class Quiet(TestCase):
    longMessage = False


class Wide(TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.failureException = KeyError


class Sub(Wide):
    maxDiff = 10


w = Wide()
print(w.describe())
w.setUp()
print(w.describe())
s = Sub()
print(s.describe())
s.setUp()
print(s.describe(), Sub.maxDiff)
q = Quiet()
print(q.describe())
q.longMessage = True
print(q.describe(), Quiet.longMessage)
items: list[TestCase] = [TestCase(), w, s, q]
print([t.maxDiff for t in items], [t.longMessage for t in items])
print(TestCase().describe(), TestCase.maxDiff)
