# WHAT: a class value -- `type(x)`, `x.__class__`, a class held in a variable,
# a container or a union -- names the class the VALUE has at run time: its
# name, qualname, module and repr, identity and equality with another class
# value, its hash, and isinstance/issubclass against it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every answer here depends
# on which class the value holds, which the static type does not decide (`x: A`
# may hold a B). Before class values carried their type object, `type(x)` on an
# open class was refused, `type(a) is type(b)` over two erased values folded to
# True, and `isinstance(e, t)` with `t: type[BaseException]` was always True.
class A:
    pass


class B(A):
    pass


def describe(x: A) -> str:
    t = type(x)
    return t.__name__ + " " + t.__qualname__ + " " + t.__module__ + " " + repr(t)


print(describe(A()), describe(B()))
print(B().__class__ is B, A().__class__ is B, type(B()) == A)


def same(a: object, b: object) -> bool:
    return type(a) is type(b)


print(same(1, 2), same(1, "s"), same([1], [2]), same(A(), B()))
print(hash(A) == hash(A), A == A, A != B, int == int, int == str)

classes = [A, B]
print([c.__name__ for c in classes], B in classes, int in [str, int])
seen = {A, B, A}
print(len(seen), A in seen, str in {int, str})
names: dict[type, str] = {int: "an int", str: "a str", list: "a list"}


def kind(o: object) -> str:
    return names.get(type(o), "something else")


print(kind(1), kind("s"), kind([1]), kind(1.5))
registry: dict[str, type[A]] = {"a": A, "b": B}
print(registry["b"].__name__, registry.get("b") is B, registry.get("z") is None)


def find(name: str) -> type[A] | None:
    if name == "b":
        return B
    if name == "a":
        return A
    return None


for n in ["a", "b", "c"]:
    c = find(n)
    print(n, "none" if c is None else c.__name__)


def check(o: object, t: type) -> bool:
    return isinstance(o, t)


print(check(B(), A), check(A(), B), check(1, int), check("s", int))


def below(sub: type, sup: type) -> bool:
    return issubclass(sub, sup)


print(below(B, A), below(A, B), below(bool, int), below(KeyError, LookupError))
print(type(None), type(None).__name__)


def classes_of(vs: list[int | str | float]) -> list[str]:
    return [type(v).__name__ for v in vs]


print(classes_of([1, "s", 2.5]))
