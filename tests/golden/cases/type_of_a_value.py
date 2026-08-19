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
# ⛔ Refused, and both refusals are the point: `type(a)` where the static class
# has a subclass (the answer would name the static class, which is what CPython
# would NOT print), and `type(o)` on a type-erased value.
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
