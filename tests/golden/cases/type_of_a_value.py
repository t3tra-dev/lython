# What this pins: `type(x)` and the two things programs do with it -- read
# `__name__` and compare it against a class with `is`.
#
#     print(type(x).__name__)     # unresolved name 'type'
#     print(type(x) is C)         # `is` requires reference-typed operands
#
# The name was simply unbound, which took the standard "what did I get" idiom
# with it. It is answerable statically exactly when nothing can put a SUBCLASS
# instance in x: a manifest contract is its own runtime class here (a bool is a
# truth bit, not an int), and a source class is too unless the program declares
# a subclass of it.
#
# The identity fold is the other half. A class has exactly one type object in
# CPython, so two type objects are the same object iff they name the same class
# -- decided here, the way `C.__name__` already was, because the answer cannot
# depend on anything the program does at run time.
#
# Why this must run: `type(f())` must CALL f, and once, which only a counter
# shows; the folded name and the folded identity are otherwise invisible to a
# reader of the IR. The union arms are here because narrowing is what makes the
# static class exact in each of them.
#
# ⭐ A UNION KNOWS ITS MEMBERS and the value carries which one it is, so
# `type(v).__name__` over `int | str | float` is the member's name selected by
# the same tag test isinstance uses -- written as the conditional expression a
# reader would have written, so the tests, the narrowing and the constants all
# come from paths that already exist. The subject has to be a NAME, because the
# chain mentions it once per member.
#
# `x.__class__` is the same question spelled as an attribute, so it takes the
# same road -- including the dynamic read for a subclassed class -- and
# `type(a) == type(b)` folds like the `is` spelling, because a class has exactly
# one type object and a reader picks either.
#
# ⛔ Refused, and the refusal is the point: `type(o)` on a type-erased value, and
# `type(v) is B` where the static class has a subclass -- the type OBJECT there
# would have to be a runtime value, which is a different mechanism from the name.
#
# ⭐ THE NAME ITSELF IS ANSWERABLE FOR A SUBCLASSED CLASS, because the instance
# header carries its class id in word 1 -- the word isinstance reads -- and the
# program's class-name table maps it. So `type(shape).__name__` over a list of
# Shape prints Circle, Square, Shape, which is what CPython prints and what the
# fold could never say.
#
# ⭐ EXCEPT FOR AN EXCEPTION, which is the commonest use of type() and the one
# the fold cannot answer: a handler's static class is the one CAUGHT and CPython
# prints the one RAISED. An exception instance carries its dynamic class id in
# its header -- the traceback and the repr already read it -- so
# `type(e).__name__` lowers to a read of that id instead of folding. A SOURCE
# exception class has no manifest method of its own, so the receiver is retyped
# to its exception ancestor first, which is what keeps a user class answering
# its own name.


class C:
    pass


class D:
    pass


def kind(v: int | str) -> str:
    if isinstance(v, int):
        return type(v).__name__
    return type(v).__name__


calls = 0


def make() -> int:
    global calls
    calls += 1
    return 5


x = C()
print(type(x).__name__, type(x) is C, type(x) is D, type(x) is not D)
print(type(5).__name__, type("a").__name__, type(5) is int, type(5) is str)
print(kind(1), kind("a"))
print(type([1]).__name__, type({}).__name__, type((1,)).__name__)
print(type(make()).__name__, calls)


def described(v: int | str | float) -> str:
    return type(v).__name__


maybe: int | None = None
print(described(1), described("a"), described(2.5), type(maybe).__name__)
print(x.__class__.__name__, x.__class__ is C, x.__class__ == C)
print(type(1) == type(2), type(1) == type("a"), type(1) != type("a"))


class Shape:
    pass


class Circle(Shape):
    pass


class Square(Shape):
    pass


def name_of(s: Shape) -> str:
    return type(s).__name__


shapes: list[Shape] = [Circle(), Square(), Shape()]
print([name_of(s) for s in shapes])


class AppError(Exception):
    pass


class NotFound(AppError):
    pass


for key in ["a", "b"]:
    try:
        if key == "b":
            raise NotFound("missing " + key)
        print(key)
    except AppError as e:
        print(type(e).__name__, e)

try:
    int("x")
except ValueError as e:
    print(type(e).__name__)
try:
    raise KeyError("k")
except LookupError as e:
    print(type(e).__name__, repr(e))
