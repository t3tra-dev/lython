# WHAT: a generic method holds an instance of a generic class, built from its
# own type parameter, in a local, and uses it; the class it builds is the
# one each call's argument solves, and a method nothing calls builds none.
#
# WHY THIS IS RUN: which class the instance tests against is a run-time
# value. The local was typed before the method was specialized, with E read
# as a class named E, and the class was emitted at that fabrication ("class
# value has no type object for builtins.E") whether or not the method ran.
class Ctx[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected

    def check(self, exc_type: type[BaseException]) -> bool:
        return issubclass(exc_type, self.expected)


class TestCase:
    def raises[E: BaseException](self, expected: type[E]) -> bool:
        context = Ctx[E](expected)
        return context.check(KeyError)

    def unused[E: BaseException](self, expected: type[E]) -> str:
        holder = Ctx[E](expected)
        return holder.expected.__name__


t = TestCase()
print(t.raises(ValueError), t.raises(LookupError), t.raises(KeyError))
