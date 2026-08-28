# WHAT: `bytes(...)` in each of its spellings -- empty, a size, a bytes-like, a
# string with an encoding, and a list of ints -- and the round trip back out.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is the BYTES,
# and every spelling builds them differently. A constructor that picked the
# wrong overload compiles and returns a bytes object of the wrong length --
# which is what happened while the selection preferred whichever overload was
# declared first: `bytes([65, 66])` answered `b''`.
#
# ⛔ THE LIST SPELLING TAKES A LIST, where CPython takes any iterable of ints.
# There is no runtime iteration protocol to consume here.
print(bytes())
print(bytes(3))
print(bytes(b"xy"))
print(bytes("ab", "utf-8"))
print(bytes([65, 66, 67]))
print(len(bytes(5)), list(bytes([1, 2])), bytes([65]) + b"Z")
print(bytes("héllo", "utf-8").decode("utf-8"))


def build(values: "list[int]") -> bytes:
    return bytes(values)


print(build([104, 105]), build([]))
print(bytes(bytes([1, 2])) == bytes([1, 2]))
