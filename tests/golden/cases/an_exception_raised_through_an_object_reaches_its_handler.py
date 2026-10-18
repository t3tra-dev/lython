# WHAT: an exception a class's `__eq__`, `__hash__`, `__repr__`, `__str__`
# or `__bool__` raises while the instance is held as `object` -- `==`, `!=`,
# `hash()`, `repr()`, `str()`, a truth test, `in` over a `list[object]`, a
# `dict[object, int]` lookup, printing a `list[object]` -- reaches the
# `except` written around it; the same calls on builtin values answer.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether the handler runs
# is the unwind at run time. `==` and `!=` walked past the `try`: the runtime
# function behind them called only the hook, a declaration until the LLVM
# layer, and a `__ly_*` declaration was taken for one that cannot raise.
class Bad:
    def __eq__(self, other: object) -> bool:
        raise KeyError("eq")

    def __hash__(self) -> int:
        raise LookupError("hash")

    def __repr__(self) -> str:
        raise ValueError("repr")

    def __str__(self) -> str:
        raise RuntimeError("str")

    def __bool__(self) -> bool:
        raise ArithmeticError("bool")


def eq(a: object, b: object) -> str:
    try:
        return str(a == b)
    except KeyError as e:
        return "eq caught " + str(e)


def ne(a: object, b: object) -> str:
    try:
        return str(a != b)
    except KeyError as e:
        return "ne caught " + str(e)


def hsh(a: object) -> str:
    try:
        return str(hash(a))
    except LookupError as e:
        return "hash caught " + str(e)


def rep(a: object) -> str:
    try:
        return repr(a)
    except ValueError as e:
        return "repr caught " + str(e)


def st(a: object) -> str:
    try:
        return str(a)
    except RuntimeError as e:
        return "str caught " + str(e)


def tr(a: object) -> str:
    try:
        return "T" if a else "F"
    except ArithmeticError as e:
        return "bool caught " + str(e)


def member(a: object, xs: list[object]) -> str:
    try:
        return str(a in xs)
    except KeyError as e:
        return "in caught " + str(e)


def lookup(a: object, d: dict[object, int]) -> str:
    try:
        return str(d.get(a))
    except LookupError as e:
        return "dict caught " + str(e)


def listing(xs: list[object]) -> str:
    try:
        return str(xs)
    except ValueError as e:
        return "list caught " + str(e)


b = Bad()
print(eq(b, 1), "|", ne(b, 1), "|", hsh(b))
print(rep(b), "|", st(b), "|", tr(b))
print(member(1, [b, 2]), "|", lookup(b, {1: 1}), "|", listing([1, b]))
print(eq(1, 1), ne("a", "b"), hsh(7), rep([1]), st(2.5), tr([]),
      member(2, [1, 2]), lookup(1, {1: 7}), listing([1, "a"]))
