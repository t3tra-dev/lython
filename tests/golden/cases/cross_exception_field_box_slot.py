# Cross-track: a user exception's own fields live in the [count, count x box16]
# block hung off extended header word 4 (Wave 2.5), and an ordinary class's
# fields now live in box16 slots on the instance (kernel/4a). The two storages
# are different mechanisms, and they meet when an exception instance is what an
# ordinary class's field HOLDS: the slot carries the exception handle, and the
# exception's own field block is reached through that handle.
#
# What that combination has to preserve: the exception caught in a handler
# survives being stored into a field and read after the handler ends; mutating
# its field through a borrowed parameter is visible to the owner; rebinding the
# field to a fresh exception is visible across the function boundary; and an
# exception reached out of a container element's field can be re-raised and
# still read afterwards.


class DecodeError(ValueError):
    def __init__(self, msg: str, pos: int) -> None:
        self.msg = msg
        self.pos = pos
        super().__init__(msg + " @ " + str(pos))


class Report:
    def __init__(self, e: DecodeError, tag: str) -> None:
        self.e: DecodeError = e
        self.tag: str = tag


def boom(text: str) -> int:
    raise DecodeError("bad", len(text))


def collect(text: str) -> Report:
    out: Report = Report(DecodeError("none", -1), "clean")
    try:
        boom(text)
    except DecodeError as caught:
        out = Report(caught, "caught")
    return out


r = collect("xyz")
print(r.tag, r.e.msg, r.e.pos)
print(str(r.e))


# A store into the exception's word-4 block reached through the holder's slot.
def retag(rep: Report, t: str) -> None:
    rep.tag = t
    rep.e.msg = t + "!"


retag(r, "moved")
print(r.tag, r.e.msg, r.e.pos)


# A rebind of the slot itself, from a callee, to a freshly raised-shape value.
def swap(rep: Report) -> None:
    rep.e = DecodeError("fresh", 7)


swap(r)
print(r.e.msg, r.e.pos, str(r.e), r.tag)

held: list[Report] = [collect("ab"), collect("cde")]
print(held[0].e.msg, held[0].e.pos, held[1].e.pos)
try:
    raise held[1].e
except DecodeError as again:
    print(again.msg, again.pos, str(again))
print(held[1].e.pos, held[1].e.msg)
