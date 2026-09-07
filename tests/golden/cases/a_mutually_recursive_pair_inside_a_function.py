# Why execution: the answers below are what must not change while the emission
# does. ⛔ THIS GOLDEN CANNOT GO RED ON THE PRE-FIX BINARY -- the defect was a
# LEAK and the printed answers were always right. What red-checks the leak is
# tests/probe/tools/leak_sweep.py, measured on this file: 1038 allocs / 83128 B
# before, 0 after. The golden's job is to keep the answers while the closure
# machinery changes under them.
# A pair of nested defs that call each other captured each other through
# the enclosing frame's cell, which made a cycle -- cell holds one function
# object, whose closure store holds the other, whose store holds the cell --
# and this runtime has no cycle collector, so every call of the enclosing
# function leaked it. The loop at the end is the part that would grow without
# bound; the printed answers are what must not change while it stops doing so.
#
# A group whose members capture nothing but each other needs no closure at all:
# each one is reachable by SYMBOL, the way a nested def already reaches itself.
# The symbols are assigned for the whole group before any member's body is
# emitted, because the pair is circular and whichever body goes first would
# otherwise name a symbol that does not exist.
#
# ⭐ A member that also reads the ENCLOSING frame is the same mechanism with a
# non-empty list: every member carries the group's captures, whether it reads
# them or not, because a member naming a sibling has to pass the sibling's
# captures and the only list it can be sure of is its own. `with_a_capture` and
# `counted_down` are here for that.
#
# ⛔ A GENERATOR member is excluded from the group: its frame is a state
# machine with its own capture seeding, and putting one in swapped a readable
# refusal for an internal one about clone entry seeding.
class Walk:
    def run(self, n: int) -> int:
        def down(k: int) -> int:
            if k == 0:
                return 0
            return up(k - 1) + 1

        def up(k: int) -> int:
            return down(k)

        return down(n)


def parity(n: int) -> str:
    def is_even(k: int) -> bool:
        if k == 0:
            return True
        return is_odd(k - 1)

    def is_odd(k: int) -> bool:
        if k == 0:
            return False
        return is_even(k - 1)

    return "even" if is_even(n) else "odd"


def three_of_them(n: int) -> int:
    def first(k: int) -> int:
        return 0 if k == 0 else second(k - 1)

    def second(k: int) -> int:
        return third(k) + 1

    def third(k: int) -> int:
        return first(k)

    return first(n)


def beside_a_free_one(n: int) -> int:
    def a(k: int) -> int:
        return 0 if k == 0 else b(k - 1)

    def b(k: int) -> int:
        return a(k) + 1

    def unrelated(k: int) -> int:
        return k * 2

    return a(n) + unrelated(n)


def with_a_capture(n: int, base: int) -> int:
    def a(k: int) -> int:
        return base if k == 0 else b(k - 1)

    def b(k: int) -> int:
        return a(k - 1) + 1

    return a(n)


def counted_down(n: int, step: int) -> int:
    seen = 0

    def down(k: int) -> int:
        nonlocal seen
        seen += 1
        return 0 if k <= 0 else up(k - step)

    def up(k: int) -> int:
        return down(k) + 1

    total = down(n)
    return total * 100 + seen


def text(n: int) -> str:
    def a(k: int) -> str:
        return "" if k == 0 else b(k - 1) + "a"

    def b(k: int) -> str:
        return a(k) + "b"

    return a(n)


def main() -> None:
    print(parity(6), parity(7), Walk().run(4))
    print(three_of_them(3), beside_a_free_one(3), with_a_capture(4, 10))
    print(counted_down(6, 2), counted_down(3, 1))
    print(text(2))
    total = 0
    for i in range(200):
        total += 1 if parity(i) == "even" else 0
        total += with_a_capture(2, i) % 3
    print(total)


main()
