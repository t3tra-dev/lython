# WHAT: a dict answers the same at every size its index table takes -- empty
# (no table of its own), one entry, the five-entry floor, and past the 128-
# and 32766-entry points where a table slot widens from one byte to two and
# from two to four -- through insert, lookup, get, pop, delete (which shifts
# the dense entries), iteration order, copy, update and clear.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the table is bytes the
# runtime reads and writes at a width it derives from the capacity; a slot
# read at the wrong width or offset compiles and finds the wrong entry, which
# only a lookup that executes can show.


def build(n: int) -> dict[int, int]:
    d: dict[int, int] = {}
    for i in range(n):
        d[i * 7] = i
    return d


def check(n: int) -> str:
    d = build(n)
    hits = 0
    for i in range(n):
        hits += d.get(i * 7, -1) == i
    misses = 0
    for i in range(n):
        misses += (i * 7 + 1) in d
    # A delete shifts the dense tail, O(n) each: only the small sizes delete.
    if n <= 300:
        for i in range(0, n, 3):
            del d[i * 7]
    after = 0
    for i in range(n):
        after += d.get(i * 7, -1) >= 0
    popped = d.pop(7, -1) if n > 1 else -1
    first = list(d.keys())[:3]
    c = d.copy()
    c[1] = 1
    u: dict[int, int] = {}
    u.update(c)
    total = sum(u.values())
    d.clear()
    return str(n) + " " + str(hits) + " " + str(misses) + " " + str(after) + " " + str(popped) + " " + str(first) + " " + str(len(c)) + " " + str(total) + " " + str(len(d)) + " " + str(d)


for n in (0, 1, 5, 6, 128, 129, 300, 32766, 32767, 40000):
    print(check(n))
e: dict[str, int] = {}
print(e, e.get("x", 0), "x" in e, e == {}, len(e.copy()))
lit = {"a": 1, "b": 2}
lit["c"] = 3
print(lit, {k: v for k, v in lit.items() if v > 1})
# Static keys stored one at a time are written by the compiled code itself,
# growing the dict as it goes -- past the first block and the five-entry floor.
s: dict[str, int] = {}
s["a"] = 1
s["b"] = 2
s["c"] = 3
s["d"] = 4
s["e"] = 5
s["f"] = 6
s["g"] = 7
s["h"] = 8
s["i"] = 9
print(s, len(s), s["i"], s.get("e", 0))
