# Calls inside loops must see fresh loop-carried arguments and their
# results' fields must read back what the callee stored. The int fast
# lane of a runtime value is only authoritative when its validity flag
# is statically true; class-field stores and sequence index reads must
# otherwise fall back to the boxed payload. Every block below silently
# mis-executed (stale first-iteration values) before the lane guards.


class Box:
    def __init__(self, iv: int) -> None:
        self._iv: int = iv

    def get(self) -> int:
        return self._iv


def make(i: int) -> Box:
    return Box(i + 100)


def at(xs: list[int], i: int) -> int:
    return xs[i]


def skip_spaces(doc: str, n: int, i: int) -> int:
    while i < n and doc[i] == " ":
        i = i + 1
    return i


class Cursor:
    def __init__(self) -> None:
        self.pos: int = 0


def scan(cur: Cursor, doc: str, start: int) -> int:
    end = skip_spaces(doc, len(doc), start)
    cur.pos = end + 1
    return end


def run_make_loop() -> None:
    i = 0
    while i < 3:
        item = make(i)
        print(item._iv, item.get())
        i = i + 1


def run_at_loop() -> None:
    xs = [10, 20, 30]
    for i in range(3):
        print(at(xs, i), xs[i])


def run_scan_store() -> None:
    cur = Cursor()
    end = scan(cur, "  }", 0)
    print(end, cur.pos)


def run_chained_results() -> int:
    # Loop variable advanced through a field read of each call result.
    k = 0
    guard = 0
    while guard < 10:
        guard = guard + 1
        if k < 4:
            item = make(k - 100)  # _iv == k
            k = item.get() + 1
    return k


run_make_loop()
run_at_loop()
run_scan_store()
print(run_chained_results())
