# WHAT: inside a loop, `if x is not None:` whose arm assigns `x` (None, or
# another value) and whose `else` leaves the loop: the next trip sees what the
# arm assigned.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was a wrong
# value on the next trip -- a loop that never ended, or a count off by many --
# which only running it shows.
#
# ⛔ The fact the test proved ("x is not None") was carried past the `if`
# because the `else` leaves it, and re-applied to the value the arm had just
# assigned: the None was unwrapped as the str it had replaced. argparse's
# `-vv` (one option string holding two flags) hit it.


def count_down(x: str | None) -> int:
    n = 0
    while True:
        if x is not None:
            n += 1
            x = x[1:]
            if not x:
                x = None
        else:
            return n


def stop_after_one(x: str | None) -> int:
    n = 0
    while n < 5:
        if x is not None:
            n += 1
            x = None
        else:
            return n
    return -1


def replace_then_stop() -> int:
    x: str | None = "q"
    n = 0
    while n < 5:
        if x is not None:
            n += 1
            x = "r" if n < 2 else None
        else:
            return n
    return -1


print(count_down("vvv"), stop_after_one("q"), replace_then_stop())
