# A method other than __init__ creates an instance attribute when it runs, as a
# unittest's setUp does: the attribute reads back what the method stored, a
# read before the method ran raises AttributeError, and neither a base's field
# nor a dataclass's fields gain a second slot from it.
#
# Why it runs: the read before the store raises at run time, and the values
# after it are the ones the methods stored.
from dataclasses import dataclass


class TestCase:
    def setUp(self) -> None:
        pass


class StackTest(TestCase):
    def setUp(self) -> None:
        self.items = [1, 2]
        self.label = "stack"

    def test_push(self) -> None:
        self.items.append(3)
        print(self.label, self.items)


t = StackTest()
try:
    print(t.items)
except AttributeError as e:
    print("AttributeError:", e)
t.setUp()
t.test_push()


class Base:
    def __init__(self) -> None:
        self.count = 0

    def show(self) -> str:
        return "count=%d" % self.count


class Sub(Base):
    def bump(self) -> None:
        self.count = self.count + 5
        self.extra = True


s = Sub()
s.bump()
print(s.show(), s.count, s.extra)


class Slot:
    def clear(self) -> None:
        self.value = None

    def fill(self, n: int) -> None:
        self.value = n


o = Slot()
o.clear()
print(o.value)
o.fill(4)
print(o.value)


@dataclass
class P:
    a: int
    b: int

    def __post_init__(self) -> None:
        self.total = self.a + self.b


p = P(1, 2)
print(p, p.total, p == P(1, 2))


class Acc:
    def run(self, xs: list[int]) -> None:
        for x in xs:
            if x > 1:
                self.last = x


a = Acc()
a.run([1, 2, 3])
print(a.last)
b = Acc()
b.run([1])
try:
    print(b.last)
except AttributeError as e:
    print("AttributeError:", e)


class Box[T]:
    def __init__(self, v: T) -> None:
        self.v = v

    def snapshot(self) -> None:
        self.old = self.v


box = Box("first")
box.snapshot()
box.v = "second"
print(box.old, box.v)
