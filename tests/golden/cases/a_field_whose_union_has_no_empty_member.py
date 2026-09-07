# Why execution: the defect was a runtime abort -- "Ly_DecRef observed
# non-positive refcount" -- from CONSTRUCTING the object, so nothing before
# the JIT could see it. Every program here is one that aborted, and each
# decodes the field rather than printing it, so a tag that named the wrong
# member would print the wrong arm instead of the right one.
#
# An instance starts at a dead value, whose union tag names a member that owns
# nothing -- `None` in every Optional. A union with no such member had no
# member to name, so the tag stayed at 0 and the store's release of the
# field's previous contents released a header the frame had zeroed.
class Box:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v


class Two:
    def __init__(self) -> None:
        self.a: "int | str" = 1
        self.b: "int | str" = "x"


class Measured:
    def __init__(self, v: "int | float") -> None:
        self.v: "int | float" = v

    def kind(self) -> str:
        seen = self.v
        if isinstance(seen, int):
            return "int:" + str(seen + 1)
        return "float"


class Raw:
    def __init__(self, v: "str | bytes") -> None:
        self.v: "str | bytes" = v


class Optional:
    def __init__(self, v: "int | None") -> None:
        self.v: "int | None" = v


class Sub(Box):
    def __init__(self) -> None:
        Box.__init__(self, 3)


def decode(v: "int | str") -> str:
    if isinstance(v, str):
        return "s:" + v
    return "i:" + str(v + 1)


shared = Box(11)


def reads_the_global() -> str:
    return decode(shared.v)


def main() -> None:
    number = Box(7)
    text = Box("ab")
    print(decode(number.v), decode(text.v))

    inner = number.v
    if isinstance(inner, int):
        print(inner * 2)

    number.v = "later"
    print(decode(number.v))

    pair = Two()
    print(decode(pair.a), decode(pair.b))

    print(Measured(4).kind(), Measured(1.5).kind())

    raw = Raw("q")
    kept = raw.v
    if isinstance(kept, str):
        print(kept + "!", len(kept))

    # The Optional spelling has always worked; it shares the tag choice, so it
    # is here to say the member that owns nothing is still the one named.
    print(Optional(5).v, Optional(None).v)

    print(decode(Sub().v), reads_the_global())

    collected: "list[int | str]" = [number.v, text.v]
    print(len(collected), decode(collected[0]))


main()
