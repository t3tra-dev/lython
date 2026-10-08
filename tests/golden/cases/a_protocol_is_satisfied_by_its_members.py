# WHAT: a typing.Protocol is satisfied by every class with its members, none
# of which names it: methods (a default parameter beyond the protocol's is
# fine), a data member answered by a field, a class attribute or a property, a
# property member answered by a field that is later reassigned, a protocol
# that extends another, a dunder member reached through len(), and a class
# that subclasses the protocol and inherits its default method. Values of the
# protocol type live in lists, dict values and an Optional field, and are
# dispatched on their own class.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which body answers is the
# runtime class of each value; a call that ran the protocol's stub compiles
# and prints None.

from typing import Protocol

class Shape(Protocol):
    def area(self) -> float: ...

class Square:
    def __init__(self, s: float) -> None:
        self.s = s
    def area(self) -> float:
        return self.s * self.s

class Circle:
    def __init__(self, r: float) -> None:
        self.r = r
    def area(self) -> float:
        return 3.0 * self.r * self.r

def total(shapes: list[Shape]) -> float:
    t = 0.0
    for s in shapes:
        t += s.area()
    return t

def one(s: Shape) -> float:
    return s.area()

print(one(Square(2.0)), one(Circle(1.0)))
print(total([Square(2.0), Circle(1.0)]))


class Named(Protocol):
    name: str

class Greeter(Named, Protocol):
    def greet(self, other: str) -> str: ...

class HasLen(Protocol):
    def __len__(self) -> int: ...

class HasKind(Protocol):
    @property
    def kind(self) -> str: ...

class Base:
    def greet(self, other: str) -> str:
        return "base greets " + other

class Person(Base):
    def __init__(self, name: str) -> None:
        self.name = name
    def greet(self, other: str, loud: bool = False) -> str:
        text = self.name + " greets " + other
        return text.upper() if loud else text

class Robot:
    name = "R2"
    def greet(self, other: str) -> str:
        return "beep " + other

class Bag:
    def __init__(self, n: int) -> None:
        self.n = n
    def __len__(self) -> int:
        return self.n
    @property
    def kind(self) -> str:
        return "bag"

class Pet:
    def __init__(self) -> None:
        self.name = "rex"

class Registry:
    def __init__(self) -> None:
        self.members: dict[str, Greeter] = {}
        self.first: Greeter | None = None
    def add(self, g: Greeter) -> None:
        self.members[g.name] = g
        if self.first is None:
            self.first = g

def hello_all(gs: list[Greeter], who: str) -> list[str]:
    return [g.greet(who) for g in gs]

def names(ns: list[Named]) -> list[str]:
    return [n.name for n in ns]

def size(s: HasLen) -> int:
    return len(s)

def kind_of(k: HasKind) -> str:
    return k.kind

r = Registry()
for g in [Person("ann"), Robot(), Person("bob")]:
    r.add(g)
print(hello_all(list(r.members.values()), "you"))
print(names([Person("x"), Robot(), Pet()]))
print(size(Bag(3)), kind_of(Bag(1)))
f = r.first
if f is not None:
    print(f.greet("first"))


class Speaker(Protocol):
    def greet(self) -> str: ...
    def twice(self) -> str:
        return self.greet() + self.greet()

class Loud(Speaker):
    def greet(self) -> str:
        return "HI" * 300

class Quiet:
    def __init__(self, word: str) -> None:
        self.word = word
    def greet(self) -> str:
        return self.word
    def twice(self) -> str:
        return "<" + self.word + ">"

def run_speakers(gs: list[Speaker]) -> list[int]:
    out: list[int] = []
    for g in gs:
        out.append(len(g.twice()))
    return out

speakers: dict[str, Speaker] = {}
for i in range(5):
    speakers[str(i)] = Loud() if i % 2 == 0 else Quiet("q" * (100 * i))
print(run_speakers(list(speakers.values())))
first_speaker = speakers.pop("0")
print(len(first_speaker.greet()), len(speakers))


class Kinded(Protocol):
    @property
    def kind(self) -> str: ...

class FieldKind:
    def __init__(self) -> None:
        self.kind = "field"

class PropKind:
    @property
    def kind(self) -> str:
        return "prop"

class AttrKind:
    kind = "attr"

def kind_of_any(x: Kinded) -> str:
    return x.kind

print([kind_of_any(FieldKind()), kind_of_any(PropKind()), kind_of_any(AttrKind())])
f = FieldKind()
f.kind = "changed"
print(kind_of_any(f))
