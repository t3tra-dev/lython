# What this pins: `list.sort()` above the block size, where the MERGE runs.
#
# The sort is minrun-sized blocks made sorted in place and then merged in
# balanced bottom-up passes, so every list shorter than minrun (32..64) leaves
# the merge untouched. The whole suite's other fifty sorts are of fifty
# elements or fewer, which is to say the merge had nothing on it.
#
# Why this needs to run: a sort's answer is a runtime value. There is no
# diagnostic, and a merge that drops, duplicates or reorders an element
# produces a list that is still a list.
#
# The sizes bracket the boundary (63, 64, 65, 129) because that is where the
# block loop's last partial block and the merge's first pass meet, and 200 for
# several passes. Every expected line is python3.14's.


class Rec:
    key: int
    tag: int

    def __init__(self, key: int, tag: int) -> None:
        self.key = key
        self.tag = tag

    def __lt__(self, other: "Rec") -> bool:
        return self.key < other.key


def scrambled(n: int) -> list[int]:
    xs: list[int] = []
    i: int = 0
    while i < n:
        xs.append((i * 7919) % 1009)
        i = i + 1
    return xs


def descending(n: int) -> list[int]:
    xs: list[int] = []
    i: int = 0
    while i < n:
        xs.append(n - i)
        i = i + 1
    return xs


def ascending(n: int) -> list[int]:
    xs: list[int] = []
    i: int = 0
    while i < n:
        xs.append(i)
        i = i + 1
    return xs


def check(xs: list[int]) -> str:
    ys: list[int] = sorted(xs)
    ordered: bool = True
    i: int = 1
    while i < len(ys):
        if ys[i] < ys[i - 1]:
            ordered = False
        i = i + 1
    return f"{len(ys)} {ordered} {sum(ys) == sum(xs)} {ys[0]} {ys[len(ys) - 1]}"


# --- the block/merge boundary ----------------------------------------------
print(check(scrambled(63)))
print(check(scrambled(64)))
print(check(scrambled(65)))
print(check(scrambled(129)))
print(check(scrambled(200)))

# --- the shapes the run detection is for -----------------------------------
print(check(descending(200)))
print(check(ascending(200)))

# --- duplicates, so equal elements meet in a merge -------------------------
dups: list[int] = []
d: int = 0
while d < 200:
    dups.append(d % 5)
    d = d + 1
print(sorted(dups)[:8], sorted(dups)[192:])

# --- STABILITY through the merge: equal keys keep their relative order ------
recs: list[Rec] = []
r: int = 0
while r < 200:
    recs.append(Rec(r % 7, r))
    r = r + 1
recs.sort()
tags: list[int] = []
for rec in recs:
    tags.append(rec.tag)
print(tags[:10])
print(tags[100:110])
print(tags[190:])

# --- a long ordered prefix followed by a scrambled tail --------------------
mixed: list[int] = []
m: int = 0
while m < 150:
    mixed.append(m)
    m = m + 1
while m < 300:
    mixed.append((m * 7919) % 601)
    m = m + 1
print(check(mixed))
