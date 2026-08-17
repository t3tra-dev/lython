# What this pins: an empty container passed to a source-class METHOD.
#
#     class C:
#         def take(self, xs: list[int]) -> int: ...
#     C().take([])
#     # argument 'xs' of 'take' is declared list[int] and this call gives it
#     # list[object]
#
# An empty literal has nothing to infer an element type from, so it needs the
# parameter's declared type as its expectation. A free function got one --
# `f([])` against `def f(xs: list[int])` has always worked, because the call
# operands distribute the callee's positional types -- and a manifest method
# got one, so `xs.extend([])` and `"".join([])` were fine. The inlined
# source-class method emitted its arguments with no expectation at all, and the
# declared-parameter check then refused the call the annotation was written for.
#
# The `set()` spelling needs the COERCION as well, and a literal does not: a
# call's result type comes from the callee contract, so the expectation reaches
# the emission and the value still comes back `set[object]`. Only an erased
# container of the SAME contract is adopted -- a genuinely different type stays
# the declared-parameter check's business, which the last section is the control
# for.
#
# Why this needs to run rather than assert on a diagnostic: the parameter type
# is what the BODY is compiled against, and an empty container that adopted the
# wrong element type shows up only where a real element is added later. The
# store below keeps what it was handed and reads it back.
#
# Every expected line is python3.14's.


class Store:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.index: dict[str, int] = {}

    def load(self, rows: list[int], index: dict[str, int]) -> int:
        self.rows = rows
        self.index = index
        return len(rows) + len(index)

    def tag(self, names: set[str], label: str = "x") -> str:
        return label + str(len(names))

    def grow(self, extra: list[int]) -> int:
        self.rows.extend(extra)
        return len(self.rows)


s = Store()

# --- the literal spellings, empty and not --------------------------------
print(s.load([], {}))
print(s.load([1, 2], {"a": 1}))
print(len(s.rows), len(s.index))

# --- the constructor spelling, which also needs the coercion -------------
print(s.tag(set()))
print(s.tag({"a", "b"}))

# --- keywords, positional-with-default, and a nested call ----------------
print(s.tag(set(), label="y"))
print(s.tag(names=set(), label="z"))
print(s.grow([]))
print(s.grow([7]))
print(s.rows)


# --- a method whose parameter is a plain type is unchanged ---------------
class Adder:
    def add(self, a: int, b: int = 10) -> int:
        return a + b


print(Adder().add(1), Adder().add(1, 2), Adder().add(b=3, a=4))


# --- THE CONTROL: a genuinely wrong type is still refused ----------------
# `Store.load` declares `list[int]`; handing it a `dict` is not an empty
# container adopting a type, and the declared-parameter check must still say
# so. That refusal is asserted in EmitterTests, not here -- this case only
# shows that the shapes above did not open a hole: a NON-empty list of the
# wrong element type is rejected at the same boundary.
print(s.load([3], {"b": 2}))
