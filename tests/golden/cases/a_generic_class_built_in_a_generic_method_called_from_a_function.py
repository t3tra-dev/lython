# WHAT: a generic method builds a generic class from its own type parameter,
# and is called from a function body. The class's `__init__` annotates a
# field with the class's parameter; its body is compiled after the class is
# declared, with the parameter bound to what the specialization says.
#
# WHY THIS IS RUN: the class E is bound to decides what `take` answers at run
# time. Compiled with nothing bound, `E` read as a class of that name and the
# store into the field was refused by the lowering.
class Ctx[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected
        self._exception: E | None = None

    def take(self, exc: BaseException) -> bool:
        if isinstance(exc, self.expected):
            self._exception = exc
            return True
        return False

    def caught(self) -> str:
        exc = self._exception
        if exc is None:
            return "nothing"
        return type(exc).__name__ + ": " + str(exc)


class Case:
    def raises[E: BaseException](self, expected: type[E]) -> Ctx[E]:
        return Ctx[E](expected)


def run() -> None:
    c = Case().raises(ValueError)
    print(c.caught())
    print(c.take(KeyError("k")), c.caught())
    print(c.take(ValueError("v")), c.caught())
    d = Case().raises(LookupError)
    print(d.take(KeyError("k")), d.caught())


run()
