# A bytes entity is ONE handle, and its payload is reached by loading a base
# address out of that handle. These shapes are the ones a lane-carrying
# representation could get right by accident: while `bytes` cached its payload
# lane beside the handle, a box, a field slot and a container element each held
# their own copy of the pointer and the length, so a reader could agree with
# CPython while reading a snapshot rather than the entity.
#
# Each case therefore reads a bytes value back through an indirection that
# reallocates AFTER the value was stored -- a container that grows past its
# capacity, a field rebound on an instance obtained from a call, a dict whose
# table rehashes. What is pinned is that the payload the reader sees is the one
# the entity owns now.


class Holder:
    def __init__(self, data: bytes) -> None:
        self.data: bytes = data


def make(data: bytes) -> Holder:
    return Holder(data)


# Container element read back after the container reallocated several times.
chunks: list[bytes] = []
i: int = 0
while i < 9:
    chunks.append(b"chunk")
    i = i + 1
print(len(chunks))
print(chunks[0])
print(chunks[8])
print(b"".join(chunks) == b"chunk" * 9)

# Field slot on a call-produced instance, rebound to a longer payload.
h = make(b"short")
print(h.data)
h.data = b"a considerably longer payload"
print(h.data)
print(len(h.data))

# Dict keyed by bytes, read back after enough insertions to rehash.
table: dict[bytes, int] = {}
table[b"alpha"] = 1
table[b"beta"] = 2
table[b"gamma"] = 3
table[b"delta"] = 4
table[b"epsilon"] = 5
table[b"zeta"] = 6
table[b"eta"] = 7
table[b"theta"] = 8
table[b"iota"] = 9
print(len(table))
print(table[b"alpha"], table[b"iota"])
print(b"gamma" in table)
print(sorted(table.values()))

# Set membership after growth: hash and equality must read the same payload.
seen: set[bytes] = set()
seen.add(b"one")
seen.add(b"two")
seen.add(b"three")
seen.add(b"four")
seen.add(b"five")
seen.add(b"six")
seen.add(b"one")
print(len(seen))
print(b"three" in seen)
print(b"nope" in seen)

# A payload derived from a method call, stored and read back: the slice and the
# concatenation each allocate a fresh entity whose payload base differs from
# the source's.
parts: list[bytes] = [b"abcdef"[0:3], b"abcdef"[3:6], b"ab" + b"cd"]
print(parts)
print(parts[0] + parts[1])
box: Holder = Holder(parts[2])
print(box.data)
print(box.data.hex())
