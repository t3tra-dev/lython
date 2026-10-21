# What: `BaseException.__str__(e)` and `BaseException.__repr__(e)` render e.args
# the way BaseException does, past the program's override, past KeyError's
# repr-of-the-key text, and when e is held as a base-typed value; and
# `int.__repr__(True)` is int's repr while `int.__str__(False)` still asks
# bool's repr, as CPython's int has no `__str__` of its own.
# Why run: the answers are text the runtime renders from the stored args.
class Fancy(Exception):
    def __str__(self) -> str:
        return "fancy"

    def __repr__(self) -> str:
        return "R"


def via_base(e: BaseException) -> str:
    return BaseException.__str__(e) + "|" + BaseException.__repr__(e)


f = Fancy("stored")
print(BaseException.__str__(f), str(f), BaseException.__repr__(f), repr(f))
print(via_base(f))
print(via_base(KeyError("k")), str(KeyError("k")))
print(via_base(ValueError()), via_base(Fancy(1, [2])), via_base(KeyError(3)))
print(int.__repr__(True), int.__str__(False), True.__repr__())
print(int.__and__(True, True), int.__format__(True, ""), int.__add__(True, 2))
print(int.__invert__(True), int.bit_length(True), str.upper("ab"))
