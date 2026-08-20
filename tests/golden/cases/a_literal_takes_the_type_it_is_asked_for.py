# A container is invariant, so `list[B]` never reaches a `list[A]` formal on
# its own: `join([B(), B()])` was refused with "call arguments do not match the
# Callable contract" while `ys: list[A] = [B(), B()]` -- the same literal under
# the same expectation, one line up -- has always compiled. The assignment
# coerces the built container and the call had no such step. Must run: the
# refusal is what regresses, but only the printed values show the ELEMENTS
# survived the retyping, and a container coerced element-by-element instead
# would have wrapped each one.


class A:
    def t(self) -> str:
        return "A"


class B(A):
    def t(self) -> str:
        return "B"


def join(xs: list[A]) -> str:
    out = ""
    for x in xs:
        out = out + x.t()
    return out


print(join([B(), B()]), join([A(), B()]), join([A(), A()]))


def by_key(d: dict[str, A]) -> str:
    out = ""
    for k in sorted(d):
        out = out + d[k].t()
    return out


print(by_key({"a": A(), "b": B()}))


def flatten(rows: list[list[A]]) -> str:
    out = ""
    for row in rows:
        for x in row:
            out = out + x.t()
    return out


print(flatten([[A(), B()], [B()]]))


def pair(p: tuple[A, A]) -> str:
    return p[0].t() + p[1].t()


print(pair((A(), B())))

# The negative control: a literal whose elements do NOT fit keeps its own type,
# so the mismatch is still reported against the formal rather than retyped
# here. `[1, "a"]` under `list[int]` stays a heterogeneous literal.
inner: list[str | None] = ["a", None]
head = inner[0]
if isinstance(head, str):
    print("union element survived:", head)
