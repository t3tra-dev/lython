# WHAT: an exception whose class declares `__str__` prints by it when held as
# a builtin base whose name is longer than a short string's inline buffer
# (`ArithmeticError`), caught by a tuple of classes or passed as that base.
#
# WHY THIS IS RUN: which `__str__` runs is the caught object's class, known
# only at run time. The gate that walks a builtin base's ancestors read them
# from a list freed before the walk; a name too long to live inside the
# string's own buffer read back as garbage, and the builtin's text (the
# message given to `__init__`) was printed with no diagnostic.
class Missing(ZeroDivisionError):
    pass


class Coded(ZeroDivisionError):
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


def as_arithmetic(e: ArithmeticError) -> str:
    return str(e)


print(describe(1))
print(describe(7))
print(as_arithmetic(Coded(3)))
