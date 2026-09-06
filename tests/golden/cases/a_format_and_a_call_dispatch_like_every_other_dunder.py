# What: `format(x, spec)`, `f"{x:spec}"` and `x(...)` on a base-typed receiver
# run the subclass's body, the way `repr(x)` and `len(x)` beside them already
# did. Both sites went straight to the refusal instead of asking the dispatcher
# first, so a hierarchy whose eleven other dunders dispatched was rejected on
# these two -- one question, three spellings, one answer missing from each.
#
# Running it is the only evidence of WHICH body ran: each class returns a
# different string, and the spec and the arguments are carried through so a
# dispatcher that dropped them would print something else rather than fail.
class Tag:
    def __format__(self, spec: str) -> str:
        return "tag[" + spec + "]"

    def __call__(self, n: int, extra: int = 0) -> int:
        return n + extra


class Loud(Tag):
    def __format__(self, spec: str) -> str:
        return "LOUD[" + spec + "]"

    def __call__(self, n: int, extra: int = 0) -> int:
        return (n + extra) * 10


tags: list[Tag] = [Tag(), Loud()]

print("format", [format(t, ">4") for t in tags])
print("empty spec", [format(t, "") for t in tags])
print("f-string", [f"{t:<3}" for t in tags])
print("plain f-string", [f"{t}" for t in tags])
print("call", [t(2) for t in tags])
print("call with a keyword", [t(2, extra=1) for t in tags])
