# WHAT: `type(a) is type(b)`, `type(x) is C` and their negations -- the exact
# class test unittest's `TestCase.__eq__` and `__hash__` are written with --
# narrow the operand on the side the test proves, and decide the test where
# the two sides' classes cannot meet; a parameter that shadows a class name
# stays a class value.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which arm runs depends on
# the run-time classes (`T("a") == U("a")` is False only because U is not T),
# and the arm a call cannot reach (`T("a") == 3`) is one CPython never runs.
# `other.n` behind the test was refused ("attr.get object type has no class
# schema") because the test narrowed nothing.
class T:
    def __init__(self, n: str) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.n == other.n

    def __hash__(self) -> int:
        return hash((type(self), self.n))


class U(T):
    pass


def name_of(o: object) -> str:
    if type(o) != T:
        return "not T"
    return o.n


print(T("a") == T("a"), T("a") == U("a"), U("b") == U("b"), T("a") == 3)
print(len({T("a"), T("a"), U("a"), T("b")}))
print(name_of(T("x")), name_of(U("y")), name_of(5))


def check(T: type[object], x: object) -> bool:
    if type(x) is T:
        return True
    return False


print(check(U, U("z")), check(T, U("z")), check(int, 3))
