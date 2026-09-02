# A `with` body that has ANY control flow after the last use of its `as` value
# is refused by the ownership verifier:
#
#     owned resource from @LyTextIO_Enter result 0 is released or transferred
#     more than once on one CFG path
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree; also on the binary from
# the start of the session, so it is not new). The ingredient is a BLOCK SPLIT
# after the handle's last use, whatever makes it:
#
#   size = len(handle.read()); total = size ............... correct
#   size = len(handle.read()); total = size + 1 ........... refused
#   size = len(handle.read()); total = size // 2 .......... refused
#   size = len(handle.read()); if flag: size = size + 1 ... refused
#   size = len(handle.read()); total = size + 1; and then
#       another `handle.read()` after it .................. correct
#   the same arithmetic OUTSIDE the with .................. correct
#   `total = n + 1` with no call at all ................... correct
#
# The split is what matters, not the operator: an `if` does it, and so does the
# guarded fast/slow diamond an int add emits when both operands carry a lane
# ([[lython-unboxed-int-lane]]). Using the handle again AFTER the split moves
# its last use past the split and the program compiles, which is the same fact
# read from the other side.
#
# ⭐ TWO RELEASES OF THE `as` VALUE REACH ONE PATH. The lowered body has one
# `LyTextIO_DecRef(%34)` at the handle's LAST USE inside the body and another
# in the block every exit funnels through (three predecessors: the normal
# path, and two unwind edges). With no split the two coincide; with one, the
# normal path takes both.
#
# ⛔ NOT a wrong answer -- the program does not compile -- and not a small fix:
# "do not release at the last use when an exit block downstream already
# releases" is the release-placement algorithm's own question, in
# Passes/Runtime/Passes/Ownership.cpp.
def sized(flag: bool) -> int:
    with open("/dev/null") as handle:
        size = len(handle.read())
        if flag:
            size = size + 1
    return size


print(sized(True))
