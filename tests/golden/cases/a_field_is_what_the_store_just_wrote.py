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
# ⭐ AND THE MERGE, which is the lazy-cache idiom's usual spelling: reading the
# field AFTER an `if` that assigned it in one arm. Both arms prove the same
# thing -- the body by the store, the fall-through by the guard's negative --
# so every path past the statement has the same answer, and a fact both arms
# AGREE on now outlives it.
#
# ⛔ Agreement and not a join. `if x is None: ...` with no store leaves None on
# one side and the payload on the other; joining those rebuilds the declared
# union, and a fact equal to the declaration is not a fact. Requiring the two
# sides to be EQUAL is what keeps a proof out of the code after an `if` that
# established nothing -- the failure the branch-local rule was written for.
# ⭐ AND THE DESUGARS SEE IT. `self.d.get(k)` with one argument is not a runtime
# method at all -- it is a rewrite gated on the receiver being dict-typed -- and
# the predicate that gate asks used the raw inference, which does not carry
# these facts. So a proved dict field fell through to the manifest path and
# died as "runtime manifest has no builtins.dict.get method", a sentence about
# a method the program is right to call.
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

    def lazy_merged(self) -> list[int]:
        if self.xs is None:
            self.xs = [4, 5, 6]
        return self.xs

    def lazy_scalar(self) -> int:
        if self.f is None:
            self.f = 12
        return self.f

    def proves_nothing(self) -> str:
        # Neither arm agrees with the other, so nothing survives the statement
        # and the read below is the whole union again.
        if self.tag is None:
            pass
        return "unset" if self.tag is None else self.tag

    def overwritten(self) -> int:
        self.f = 1
        self.f = 2
        return self.f

    def cleared(self) -> str:
        self.f = 3
        self.f = None
        return "none" if self.f is None else "value"


class Cache:
    def __init__(self) -> None:
        self.hot: "dict[str, int] | None" = None
        self.cold: "dict[str, int] | None" = None

    def get(self, k: str) -> "int | None":
        if self.hot is None:
            self.hot = {}
        found = self.hot.get(k)
        if found is not None:
            return found
        if self.cold is None:
            self.cold = {"x": 9}
        deep = self.cold.get(k)
        if deep is not None:
            self.hot[k] = deep
        return deep

    def size(self) -> int:
        if self.hot is None:
            self.hot = {}
        return len(self.hot)


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
    merged = Box()
    print(merged.lazy_merged(), merged.lazy_merged(), merged.lazy_scalar())
    print(merged.proves_nothing())
    print(through_a_local_name())
    cache = Cache()
    print(cache.get("x"), cache.get("x"), cache.get("zz"), cache.size())


main()
