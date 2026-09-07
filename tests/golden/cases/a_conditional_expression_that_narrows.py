# The narrowing an `isinstance` proves, spent inside a conditional EXPRESSION.
# The `if` statement spelling of each of these compiles; the one-line spelling
# used to be refused, because the expression knew how to unwrap a union member
# and nothing else.
#
# ⭐ AND THE SAME EXPRESSION AS A COMPREHENSION'S ELEMENT, which is where the
# two channels last disagreed: the EMITTER narrowed the arms and the inference
# walk did not, so `[v if isinstance(v, str) else str(v) for v in xs]` built a
# list of str and typed as `list[int | str]` -- "function is annotated to
# return list[builtins.str]" for a comprehension that returns exactly that.
# The walk had a narrowing for `x is None` only, with a note that nothing had
# been measured to need the others; the comprehension is that reader.
#
# ⛔ The walk narrows into a copy of the CONTEXT's local symbols, not onto the
# scope: a comprehension target lives in the context, which wins over the scope
# when a name is resolved, so the `is None` form had the same defect and was
# silently doing nothing there too.


class Shape:
    pass


class Square(Shape):
    sides = 4

    def area(self, n: int) -> int:
        return n * n


class Circle(Shape):
    sides = 0


def area_of(s: Shape) -> int:
    return s.area(3) if isinstance(s, Square) else -1


def sides_of(s: Shape) -> int:
    return -1 if not isinstance(s, Square) else s.sides


shapes: list[Shape] = [Shape(), Square(), Circle()]
print([area_of(s) for s in shapes])
print([sides_of(s) for s in shapes])
print([(s.sides if isinstance(s, Square) else 0) for s in shapes])


# The union spelling the expression already handled must keep working, and the
# name must be its old self on the other side of the expression.
def label(v: int | None) -> str:
    out = str(v * 2) if v is not None else "none"
    return out + "/" + ("some" if v is not None else "empty")


print(label(3), label(None))


def texts(xs: "list[int | str]") -> list[str]:
    return [v if isinstance(v, str) else str(v) for v in xs]


def defaults(xs: "list[int | None]") -> list[int]:
    return [v if v is not None else 0 for v in xs]


def unique(xs: "list[int | str]") -> set[str]:
    return {v if isinstance(v, str) else str(v) for v in xs}


def keyed(xs: "list[int | str]") -> dict[str, int]:
    return {(v if isinstance(v, str) else str(v)): 1 for v in xs}


def widths(xs: "list[int | str]") -> int:
    return sum(len(v if isinstance(v, str) else str(v)) for v in xs)


def either(v: "int | str | None") -> str:
    return "n" if v is None else (v if isinstance(v, str) else str(v))


def both(v: "int | str", w: "int | str") -> str:
    return v + w if isinstance(v, str) and isinstance(w, str) else "?"


mixed: "list[int | str]" = [1, "a", 22]
print(texts(mixed), defaults([1, None, 2]))
print(sorted(unique(mixed)), sorted(keyed(mixed).items()), widths(mixed))
print(either(None), either(3), either("z"), both("a", "b"), both(1, "b"))
