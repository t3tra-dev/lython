# WHAT: int's bytes and ratio methods answer as CPython's: to_bytes and
# from_bytes in either byte order, signed and not, with the keywords spelled or
# not, round-tripping values past 64 bits; bit_count, as_integer_ratio,
# is_integer and conjugate; and every refusal (too big, negative unsigned, a
# bad byte order, a negative length) in CPython's words. `from_bytes(b,
# "little", signed=True)` also proves the classmethod is picked by its
# arguments: the first overload takes the bytes alone and read every call as
# big-endian unsigned.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the bytes
# and integers the runtime builds; an overload picked wrongly compiles and
# prints a different number.

print((5).conjugate(), (5).is_integer(), (-7).as_integer_ratio(), (0).as_integer_ratio())
print((10).bit_count(), (-255).bit_count(), (2**100 - 1).bit_count(), (0).bit_count())
print((1024).to_bytes(2, "big"), (1024).to_bytes(2, "little"), (5).to_bytes())
print((255).to_bytes(1, "big"), (-1).to_bytes(2, "big", signed=True), (-128).to_bytes(1, "big", signed=True))
print((-129).to_bytes(2, "little", signed=True), (2**70).to_bytes(10, "big"), (0).to_bytes(0, "big"))
print((300).to_bytes(length=2, byteorder="little"), (65535).to_bytes(2, byteorder="big", signed=False))
print(int.from_bytes(b"\x04\x00", "big"), int.from_bytes(b"\x04\x00", "little"), int.from_bytes(b"\xff", "big", signed=True))
print(int.from_bytes(b"\xff\xff", "big", signed=True), int.from_bytes(b"\x80\x00", "big", signed=True), int.from_bytes(b""))
print(int.from_bytes(b"\x01" * 20, "big"), int.from_bytes(b"\x80" + b"\x00" * 15, "big", signed=True))
print(int.from_bytes(bytes=b"\x01\x02", byteorder="little"), (2**64).to_bytes(9, "big"))
for v in [0, 1, -1, 127, -128, 2**63, -2**63, 2**100 + 12345]:
    n = (v.bit_length() + 8) // 8
    print(int.from_bytes(v.to_bytes(n, "little", signed=True), "little", signed=True) == v)
def tryit(k: int) -> None:
    try:
        if k == 0:
            print((256).to_bytes(1, "big"))
        elif k == 1:
            print((-1).to_bytes(1, "big"))
        elif k == 2:
            print((128).to_bytes(1, "big", signed=True))
        elif k == 3:
            print((-129).to_bytes(1, "big", signed=True))
        elif k == 4:
            print((1).to_bytes(1, "middle"))
        elif k == 5:
            print((1).to_bytes(-1, "big"))
        elif k == 6:
            print(int.from_bytes(b"x", "BIG"))
    except (OverflowError, ValueError) as e:
        print(type(e).__name__, e)
for k in range(7):
    tryit(k)

def enc(n: int, width: int) -> bytes:
    return n.to_bytes(width, "little", signed=True)

def roundtrip(xs: list[int]) -> int:
    total = 0
    for x in xs:
        b = enc(x, 16)
        total += int.from_bytes(b, "little", signed=True) - x
        total += x.bit_count() - bin(x).count("1")
    return total

def ratios(fs: list[float]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for f in fs:
        out.append(f.as_integer_ratio())
    return out

print(roundtrip([i * 7919 - 50000 for i in range(200)] + [2**100, -2**100]))
n, d = (0.375).as_integer_ratio()
print(n + d, n * d)
print(ratios([0.5, 2.25, -8.0]))
acc = 0
for i in range(1000):
    acc += i.bit_count()
print(acc)
print(sum(int.from_bytes(k.to_bytes(4, "big"), "big") for k in range(100)))
h = (2.5).hex()
print(h, float.fromhex(h), len(h))
