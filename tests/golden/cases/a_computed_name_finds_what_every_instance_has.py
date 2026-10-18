# WHAT: `getattr(o, name)(...)` with a computed name answers object's own
# attributes as CPython does -- `__init__` on a class that declares none
# returns None (and refuses an argument with CPython's sentence), and a
# `__str__`, `__repr__` or `__hash__` that only a subclass declares is
# object's for an instance of the base, through getattr and through `repr()`,
# `str()` and `hash()` alike.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which arm answers is
# decided by the name and the receiver's class at run time. A class with no
# `__init__` of its own refused every computed call ("inherits
# builtins.object.__init__, which Lython does not implement"), and the base
# instance fell past the subclass's arm to "'Base' object has no attribute
# '__str__'".
class T:
    def test_a(self) -> None:
        print("a ran")


t = T()
for name in ["test_a", "__init__"]:
    print(name, getattr(t, name)())


def init_with(o: T, name: str) -> None:
    try:
        getattr(o, name)(1)
    except TypeError as e:
        print(e)


init_with(t, "__init__")


class Base:
    def __init__(self) -> None:
        self.n = 1


class Sub(Base):
    def __str__(self) -> str:
        return "Sub!"

    def __repr__(self) -> str:
        return "Sub()"

    def __hash__(self) -> int:
        return 7


def call(o: Base, name: str) -> str:
    result = getattr(o, name)()
    if isinstance(result, int):
        return "int " + str(result == 7)
    return str(result)


print(call(Sub(), "__str__"), call(Sub(), "__repr__"), call(Sub(), "__hash__"))
print(call(Base(), "__str__")[:15], call(Base(), "__repr__")[:15],
      call(Base(), "__hash__"))


def spelled(o: Base) -> str:
    return repr(o)[:15] + " " + str(o)[:15] + " " + str(hash(o) == 7)


print(spelled(Sub()), "|", spelled(Base()))
