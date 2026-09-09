# A constructor's parameters are where a field's type comes from, and three of
# the five kinds were not read: keyword-only, `*args` and `**kwargs`. So
#
#     class A:
#         def __init__(self, *rest: int) -> None:
#             self.rest = rest
#         def size(self) -> int:
#             return len(self.rest)
#     # builtins.object does not provide manifest method '__len__'
#
# and `self.tag = tag` after `*, tag: str = "t"` declared a field nothing could
# be read out of. Writing the annotation on the ASSIGNMENT instead
# (`self.rest: "tuple[int]" = rest`) always worked, which is what says the
# parameter was the half that was not read.
#
# Why execution: the field's declared type decides what may be read back out,
# so the program has to DECODE what it stored -- the lengths, the sums and the
# concatenations below are the assertions.
#
# ⭐ The packed ones carry their annotation's CONTAINER, not the annotation:
# `*rest: int` binds `rest` to `tuple[int]` and `**kw: int` to
# `dict[str, int]`, which is what the parameter is worth inside the body.


class Options:
    def __init__(self, name: str, *values: int, tag: str = "t",
                 **extra: str) -> None:
        self.name = name
        self.values = values
        self.tag = tag
        self.extra = extra

    def total(self) -> int:
        n = 0
        for v in self.values:
            n += v
        return n

    def label(self) -> str:
        return self.name + ":" + self.tag + ":" + str(len(self.values))

    def extras(self) -> "list[str]":
        out: "list[str]" = []
        for k in sorted(self.extra.keys()):
            out.append(k + "=" + self.extra[k])
        return out


a = Options("a", 1, 2, 3)
print(a.label(), a.total(), a.values[0] + 1)
b = Options("b", tag="z", left="1", right="2")
print(b.label(), b.total(), b.extras())
print(b.extra["left"] + "!")
