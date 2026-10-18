# WHAT: `str(e)` for an exception caught by a tuple of classes calls the
# `__str__` the caught class declares.
#
# WHY THIS IS RUN: the name the handler binds is typed by the classes' common
# builtin base, so which `__str__` runs is the caught object's class, known
# only at run time. The builtin's text was printed instead -- the message
# given to `__init__` -- with no diagnostic.
class Missing(Exception):
    pass


class Coded(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("code " + str(code))
        self.code = code

    def __str__(self) -> str:
        return "Coded[" + str(self.code) + "]"


def describe(k: int) -> str:
    try:
        if k == 1:
            raise Missing("one")
        raise Coded(k)
    except (Missing, Coded) as e:
        return type(e).__name__ + ": " + str(e)


print(describe(1))
print(describe(7))
