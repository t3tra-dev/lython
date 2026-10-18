# WHAT: a program exception that declares `__str__` or `__repr__` prints by
# them however it is held -- typed as a builtin base, caught by a tuple of
# classes, kept in a list, passed as an object -- and one that declares
# neither prints as its builtin base does, under its own name.
#
# WHY THIS IS RUN: each text is chosen by the exception's class at run time:
# the static type names a builtin base, and a list or an object holds the
# exception through a box that says only "an exception".
class Coded(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("code " + str(code))
        self.code = code

    def __str__(self) -> str:
        return "Coded[" + str(self.code) + "]"

    def __repr__(self) -> str:
        return "Coded(" + str(self.code) + ")"


class Plain(Exception):
    pass


def as_exception(e: Exception) -> str:
    return str(e) + " | " + repr(e) + " | " + f"{e}"


def as_base(e: BaseException) -> str:
    return str(e) + " | " + repr(e)


def as_object(o: object) -> str:
    return str(o) + " | " + repr(o)


held: list[Exception] = [Coded(1), Plain("p"), KeyError("k"), ValueError("v")]
for e in held:
    print(as_exception(e))
    print(as_base(e))
    print(as_object(e))
    print(e)

for k in range(2):
    try:
        if k == 0:
            raise Coded(2)
        raise Plain("q")
    except (Coded, Plain) as caught:
        print("caught", caught, repr(caught))

errs: list[Exception] = [Coded(3), Plain("r"), KeyError("k2")]
print(errs)
things: list[object] = [Coded(4), 5, Plain("s")]
print(things)
