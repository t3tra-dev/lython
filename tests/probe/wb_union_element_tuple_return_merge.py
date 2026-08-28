# FIXED 2026-08-29. A tuple with a UNION element, unpacked -- ONE unpack of the
# str member was enough. Aborted in `Ly_IncRef observed non-positive refcount`.
#
# ⛔ THE UNION IS REQUIRED and so is the STR member: `tuple[str, int]` is fine,
# `tuple[str, str]` is fine, a BARE `int | str` return with both members is
# fine, and the same tuple returning the INT member is fine. What separates
# them is the LANE COUNT -- a str is two lanes read out of the box by
# arithmetic on an address, and the release planner does not read that as a use
# of the container. It placed the tuple's release BETWEEN the union build and
# the retain, so the retain ran on a string already at zero.
#
# The repair is the pair the non-union read has had all along: retain the
# element (per MEMBER, unconditionally -- an inactive member's lanes are the
# immortal placeholder whose count is saturated) and then PIN the container
# past the retain with an explicit `__len__` use.
#
# ⭐ THE PIN WENT INTO THE WRONG BRANCH FIRST and the program kept crashing,
# which read as "the pin does not help" rather than "the pin is not there". A
# repair that changes nothing is a question about whether it ran.
#
# This is what blocked the union STORE: with the read broken, storing a union
# made more programs reach the crash, and one of them ANSWERED instead --
# `d["x"] = v; d["y"] = v2` printed ('y', 'y') where CPython says ('y', 'z').
def make(n: int) -> "tuple[str, int | str]":
    if n == 0:
        return ("k", 1)
    return ("k", "z")


k, v = make(0)
print(k, v)
k2, v2 = make(1)
print(k2, v2)
