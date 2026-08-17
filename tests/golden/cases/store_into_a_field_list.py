# What this pins: storing into, and deleting from, a list held in a field.
#
#     class Box:
#         def __init__(self) -> None:
#             self.items: list[int] = []
#     b = Box()
#     b.items.append(1)
#     b.items[0] = 9
#     # IndexError: list assignment index out of range
#
# The store consulted the container's compile-time element evidence for its
# bounds check, and a field read deliberately strips the contents it knew
# (`bindRetainedEvidenceBundle` does not let them cross the read) while leaving
# the "evidence-backed" FLAG set. Zero recorded elements read as length zero, so
# an in-range store raised. `del b.items[0]` had it too, and a field SEEDED with
# `[0]` and grown by one append took the evidence arm with stale contents and
# double-booked the slot instead ("owned resource ... is released or transferred
# more than once").
#
# Both are the same rule: the evidence tier is sound only where the walk sees
# EVERY mutation of the container, and through a field it cannot -- each read
# builds a fresh bundle from the owner. So an interior view stores through the
# payload, which is authoritative. A local keeps the evidence arm, and the last
# section is its control.
#
# Why this needs to run rather than assert on a diagnostic: half of the failure
# was a raised IndexError at exit 1 and half was a refusal, but what is being
# pinned is which element the store landed on. A store that went to the payload
# at the wrong offset compiles and prints a plausible list, so every section
# reads the whole list back and the lengths with it.
#
# Every expected line is python3.14's.


class Box:
    def __init__(self) -> None:
        self.items: list[int] = []
        self.names: list[str] = []

    def put(self, i: int, v: int) -> None:
        self.items[i] = v


class Seeded:
    def __init__(self) -> None:
        self.items: list[int] = [0]


# --- the empty field grown by append --------------------------------------
b = Box()
b.items.append(1)
b.items.append(2)
b.items.append(3)
b.items[1] = 9
print(b.items, len(b.items), b.items[1])
b.items[0] = 8
b.items[2] = 7
print(b.items)
b.items[-1] = 6
print(b.items)


# --- through a method, and with a runtime index ---------------------------
b.put(1, 5)
print(b.items)
idx = 2
b.items[idx] = 4
print(b.items, b.items[idx])


# --- str elements, where the slot holds a reference -----------------------
b.names.append("a")
b.names.append("b")
b.names[0] = "z"
print(b.names, len(b.names))
b.names[1] = b.names[0]
print(b.names)


# --- the delete path, which mis-raised the same way -----------------------
d = Box()
d.items.append(1)
d.items.append(2)
d.items.append(3)
del d.items[0]
print(d.items, len(d.items))
del d.items[-1]
print(d.items, len(d.items))


# --- a field SEEDED with a literal, then grown ---------------------------
s = Seeded()
s.items.append(1)
s.items[1] = 9
print(s.items)
s.items[0] = 8
print(s.items, len(s.items))
del s.items[0]
print(s.items)


# --- an out-of-range store still raises ----------------------------------
e = Box()
e.items.append(1)
try:
    e.items[5] = 9
except IndexError as err:
    print("caught:", err)
print(e.items)


# --- THE CONTROL: a local list, which keeps the evidence arm -------------
xs: list[int] = []
xs.append(1)
xs.append(2)
xs[1] = 9
print(xs, len(xs))
ys = [0, 0]
ys[1] = 7
print(ys)
del ys[0]
print(ys)
