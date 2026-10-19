# WHAT: a generic method and a generic function test their arguments with
# `isinstance(a, ...) and isinstance(b, ...)`, `or`, and `not` of one, and
# each specialization keeps only the arm its type can reach -- including an
# arm that would not compile at the other specializations. An operand whose
# answer is known only at run time still runs, in order.
#
# WHY THIS IS RUN: which arm runs, and whether the run-time operand is
# evaluated, are seen in the output. The `and` spelling kept the dead arm
# compiled and was refused for passing an int where the arm wants a list;
# the same test as two nested `if`s was pruned.
class T:
    def listy(self, a: list[int], b: list[int]) -> str:
        return "list " + str(a == b)

    def check[V](self, first: V, second: V) -> str:
        if isinstance(first, list) and isinstance(second, list):
            return self.listy(first, second)
        return "other " + repr(first)


def shout(x: str) -> str:
    return x.upper()


def kind[V](a: V, b: V) -> str:
    if isinstance(a, str) or isinstance(b, str):
        return shout(a) if isinstance(a, str) else "?"
    if not isinstance(a, int):
        return "not int"
    return "int " + str(a)


calls: list[str] = []


def note(text: str) -> bool:
    calls.append(text)
    return True


def both[V](a: V) -> str:
    if note("first") and isinstance(a, str):
        return "str " + a
    return "other"


t = T()
print(t.check(1, 2), t.check([1, 2], [1, 2]), t.check("a", "b"))
print(kind("a", "b"), kind(3, 4), kind(2.5, 1.5))
print(both("x"), both(1), calls)
