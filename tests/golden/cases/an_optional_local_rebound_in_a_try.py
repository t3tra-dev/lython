# WHAT: a `T | None` local reassigned inside a `try`, read after it -- the
# retry loop, and the "did anything succeed" flag that every one of them has.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is which
# assignment the read observes, and which one ran is a runtime fact. Reading
# the PRE-try value compiles and prints a plausible None.
#
# ⛔ THIS WAS AN ERRORS GOLDEN. A local rebound inside a `try` is promoted into
# a one-field object so the handler and the continuation observe the
# reassignment, and a union had no contract-typed slot to be promoted into. An
# OPTIONAL does now -- it is ONE box whose empty entity word IS the None -- so
# the refusal is gone and the answer is CPython's. A WIDER union keeps its lanes
# inline, has no slot, and is still refused.
#
# ⛔ AND IT TAKES THE ORDINARY FIELD PATH, not the cell one: the cell-specific
# read has a rank-1 shape check that an optional's tag fails.
value: "int | None" = None
try:
    value = 7
except ValueError:
    pass
if value is None:
    print("none")
else:
    print(value)


class Flaky:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    def run(self) -> int:
        self.calls += 1
        if self.calls < 3:
            raise ValueError("not yet")
        return self.calls


def attempt(target: Flaky, limit: int) -> "int | None":
    result = None
    tries = 0
    while tries < limit:
        tries += 1
        try:
            result = target.run()
            break
        except ValueError:
            continue
    return result


print(attempt(Flaky(), 5), attempt(Flaky(), 2))


def first_word(lines: "list[str]") -> "str | None":
    found = None
    for line in lines:
        try:
            found = line.split()[0]
            break
        except IndexError:
            continue
    return found


print(first_word(["", "  ", "a b"]), first_word([""]))
