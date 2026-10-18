# WHAT: `getattr(o, name)(...)` and `hasattr(o, name)` with `name` computed
# answer from the class `o` has at run time: a method (inherited, overridden,
# static, class), an attribute a subclass alone declares, a field a method
# stores later, a property or a value that is not callable (CPython's
# TypeError, naming the value's class), a method the arguments do not fit
# (CPython's arity sentence), object's own names, and a name nothing declares
# (CPython's AttributeError).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the name and the class are
# both run-time values. Both lookups were refused ("getattr() needs a literal
# attribute name", "hasattr() needs a literal attribute name").
class Base:
    kind = "base"

    def __init__(self) -> None:
        self.label = "b"

    def hello(self) -> str:
        return "hello from " + type(self).__name__

    def helper(self) -> int:
        return 1

    def two(self, a: int, b: int) -> int:
        return a + b

    def opt(self, a: int = 1) -> int:
        return a

    @staticmethod
    def stat() -> str:
        return "static"

    @classmethod
    def clsm(cls) -> str:
        return "cls " + cls.__name__

    @property
    def prop(self) -> int:
        return 7


class Left(Base):
    def hello(self) -> str:
        return "left hello"

    def only_left(self) -> str:
        return "only left"


class Right(Base):
    def only_right(self) -> str:
        self.late = 1
        return "only right"


def call(o: Base, name: str) -> str:
    try:
        result = getattr(o, name)()
        if name == "__repr__":
            return "ok repr"
        if name == "__class__":
            return "ok a new " + type(result).__name__
        return "ok " + repr(result)
    except AttributeError as e:
        return "AttributeError: " + str(e)
    except TypeError as e:
        return "TypeError: " + str(e)


names = ["hello", "helper", "two", "opt", "stat", "clsm", "prop", "kind",
         "label", "only_left", "only_right", "late", "missing", "__repr__",
         "__class__", "__doc__", "__dict__"]
for o in [Base(), Left(), Right()]:
    print(type(o).__name__)
    for n in names:
        print("  ", n, "->", call(o, n), "| hasattr", hasattr(o, n))
x = "two"
print(getattr(Base(), x)(1, 2), getattr(Left(), "op" + "t")(5))
try:
    getattr(Base(), "hel" + "per")(1)
except TypeError as e:
    print(e)
