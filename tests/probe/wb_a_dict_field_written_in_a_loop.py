# FIXED 2026-09-05. All eight cells of the grid below agree with CPython now,
# and the golden is tests/golden/cases/a_dict_field_written_in_a_loop.py.
#
# What it was: `obj.field[key] = value` inside a LOOP, where the field is a
# dict built NON-EMPTY, SEGFAULTED with no diagnostic. The grid was
# {empty, non-empty} x {loop, two flat stores} x {new key, existing key}:
#   nonempty + loop + NEW key ....... SEGV
#   nonempty + loop + EXISTING key .. the affine verifier caught it
#                                     ("owned resource from @LyLong_FromI64
#                                      ... released or transferred more than
#                                      once on one CFG path")
#   the other six ................... correct
#
# ⭐ WHAT LOCATED IT: the IR. In the loop body the replace released `%8` --
# `LyLong_FromI64(1)`, the value stored at CONSTRUCTION -- with
# `aggregate_release = "builtins.int:dict.setitem"`, on every trip. The
# contents evidence describing the field was recorded where the field was
# BUILT, and a field READ produces a fresh SSA value in the loop's own block,
# so `crossesStorageDefiningBlock` compared that fresh value's block with the
# store's, found them equal, and called the evidence block-local.
#
# ⛔ The demotion is gated three ways, and each gate is a measurement:
#   only when the op's block is in a CYCLE  -- straight-line field reads have
#     evidence exactly as good as the field's;
#   only for MAPPING evidence -- demoting a list field's sequence evidence in a
#     loop takes `b.items.append(i)` from correct to "list.append on a field or
#     borrowed list is not supported inside a branch or loop body";
#   only when the field's OWNER is defined in another block, which is what
#     makes the evidence older than the trip.
#
# Kept as a probe because the grid is the thing worth re-running: the two
# failing cells differed only in the key, and one of them was caught by the
# verifier while the other crashed.


class Bag:
    def __init__(self) -> None:
        self.table: dict[str, int] = {"a": 1}


def go() -> None:
    b = Bag()
    for i in range(2):
        b.table["c"] = i
    print(sorted(b.table.items()), len(b.table))


go()
