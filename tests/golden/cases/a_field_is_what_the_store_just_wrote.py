# Why execution: every line is a value that only exists if the field was read
# back as what the store put in it. The programs were REFUSED --
#
#     self.f: "int | None"
#     self.f = 5
#     return self.f + 1
#     # static type union<int, None> does not provide manifest method '__add__'
#
# -- because a union field's read always answered the whole union, whatever the
# statement above it had just written. A store proves what it wrote in exactly
# the way a guard proves what it tested, so it is spent the same way: at the
# READ, with a check, which is what covers anything that changes the field in
# between.
#
# ⛔ The owner must be a NAME and the written type a strict MEMBER of the
# declared union -- the same two limits every other member path has. Writing
# `None` proves nothing worth keeping, and `self.f = other_union` proves the
# union it already had.
#
# ⛔ Still refused, and it is the MERGE and not the store: reading the field
# after an `if` that assigned it in one arm. Both arms prove the same thing
# there -- the body by this store and the fall-through by the guard's negative
# -- but a branch's proof does not outlive the statement, for the reason
# emitIf records. `x_lazy_else` below is the spelling that works: return inside
# the arm.
class Box:
    def __init__(self) -> None:
        self.f: "int | None" = None
        self.xs: "list[int] | None" = None
        self.tag: "str | None" = None

    def set_and_add(self) -> int:
        self.f = 5
        return self.f + 1

    def set_and_return(self) -> int:
        self.f = 7
        return self.f

    def set_and_measure(self) -> int:
        self.xs = [1, 2, 3]
        return len(self.xs)

    def set_and_join(self) -> str:
        self.tag = "ab"
        return self.tag + "!"

    def lazy(self) -> list[int]:
        if self.xs is None:
            self.xs = [9, 9]
            return self.xs
        return self.xs

    def overwritten(self) -> int:
        self.f = 1
        self.f = 2
        return self.f

    def cleared(self) -> str:
        self.f = 3
        self.f = None
        return "none" if self.f is None else "value"


def through_a_local_name() -> int:
    b = Box()
    b.f = 11
    return b.f * 2


def main() -> None:
    box = Box()
    print(box.set_and_add(), box.set_and_return(), box.set_and_measure())
    print(box.set_and_join(), box.overwritten(), box.cleared())
    fresh = Box()
    print(fresh.lazy(), fresh.lazy())
    print(through_a_local_name())


main()
