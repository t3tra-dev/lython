# WHAT: a base class's bool and class-value fields, written and read through
# base-typed functions, on instances of subclasses that add fields of their
# own -- boxed ones before which the base's used to move -- and through the
# subclass's own type.
#
# WHY THIS IS RUN: the field is one slot whatever class the instance has, and
# only reading it back shows which slot a write reached. The body kept every
# box first and the bool and class-value words after them, so a subclass
# that added a box moved its base's words: a base-typed write landed in one
# word and the subclass's read came from another, silently.
class Base:
    def __init__(self) -> None:
        self.flag = False
        self.kind: type[Exception] = ValueError
        self.name = "b"


class Sub(Base):
    def __init__(self) -> None:
        super().__init__()
        self.extra = "x"
        self.more = [1]
        self.done = True


class Leaf(Sub):
    def __init__(self) -> None:
        super().__init__()
        self.label = "leaf"


def set_flag(b: Base, value: bool) -> None:
    b.flag = value


def set_kind(b: Base, kind: type[Exception]) -> None:
    b.kind = kind


def show(b: Base) -> str:
    return str(b.flag) + " " + b.kind.__name__ + " " + b.name


for item in [Base(), Sub(), Leaf()]:
    print(show(item), end=" | ")
    set_flag(item, True)
    set_kind(item, KeyError)
    print(show(item), item.flag, item.kind.__name__)
s = Sub()
set_flag(s, True)
s.done = False
print(s.flag, s.done, s.extra, s.more, show(s))
