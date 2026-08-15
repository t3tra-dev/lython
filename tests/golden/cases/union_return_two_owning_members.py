# What this pins: a union RETURN whose members are two or more separately
# owned entities. Each member's lane is declared owned on its own, so the
# caller releases whichever one the tag says is active and leaves the rest --
# which are immortal dead placeholders -- alone.
#
# Why this needs to run rather than assert on a diagnostic: the refusal it
# replaces was a diagnostic, but what the repair has to get right is which
# lane is released, and the wrong answer there is a double free or a leak
# rather than a message. The loops below run each arm thousands of times so a
# per-iteration imbalance shows up as growth; this case is also in
# LYTHON_LEAK_GATE_CASES, where the count is checked rather than eyeballed.
#
# Every expected line is python3.14's.

# --- three members, two of them owning --------------------------------------
def pick3(n: int) -> int | str | None:
    if n == 0:
        return 5
    if n == 1:
        return "five"
    return None


print(pick3(0), pick3(1), pick3(2))


# --- four members, three of them owning, one a container --------------------
def pick4(n: int) -> int | str | list[int] | None:
    if n == 0:
        return 7
    if n == 1:
        return "seven"
    if n == 2:
        return [7, 8]
    return None


print(pick4(0), pick4(1), pick4(2), pick4(3))


# --- no None member at all: every member owns -------------------------------
def two(n: int) -> str | list[int]:
    if n == 0:
        return "a"
    return [1, 2]


print(two(0), two(1))


# --- the active member is narrowed and USED by the caller -------------------
total = 0
words = 0
i = 0
while i < 3000:
    v = pick3(i % 3)
    if v is None:
        pass
    i += 1
print("looped", i)


def narrow(n: int) -> int:
    v = pick3(n)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return len(v)
    return -1


print(narrow(0), narrow(1), narrow(2))


# --- a union return flowing straight into another union return --------------
def relay(n: int) -> int | str | None:
    return pick3(n)


print(relay(0), relay(1), relay(2))


# --- and one that only ever takes the owning arms ---------------------------
j = 0
kept = 0
while j < 2000:
    w = two(j % 2)
    if isinstance(w, str):
        kept += 1
    j += 1
print(kept)
