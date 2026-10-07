# WHAT: a bytearray refuses every resize while a memoryview holds its payload
# -- pop, remove, del of an item, a slice or an extended slice, a shrinking or
# growing slice assignment, insert, extend, +=, *=, clear -- with CPython's
# BufferError and without changing a byte, while same-size writes go through;
# the refusal comes after "value not found" and covers an empty extended
# slice; releasing every view (a slice's too) lets it resize again; an empty
# bytearray may clear while viewed; and an iterator over a released view
# raises.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the exports are counted at
# run time by the views that exist then, and the refusal has to leave the
# bytes as they were -- which only the printed bytes after it show.

b = bytearray(b"abcdef")
m = memoryview(b)
names: list[str] = []
def attempt(b: bytearray, name: str, n: int) -> None:
    try:
        if n == 0:
            b.pop()
        elif n == 1:
            b.remove(97)
        elif n == 2:
            del b[0]
        elif n == 3:
            del b[1:3]
        elif n == 4:
            del b[::2]
        elif n == 5:
            b[0:2] = b"x"
        elif n == 6:
            b[0:1] = b"xyz"
        elif n == 7:
            b.insert(0, 65)
        elif n == 8:
            b.extend(b"zz")
        elif n == 9:
            b += b"!"
        elif n == 10:
            b *= 2
        elif n == 11:
            b.clear()
        print(name, "ok", b)
    except BufferError as e:
        print(name, "BufferError", e, b)
for i, name in enumerate(["pop", "remove", "del item", "del slice", "del ext", "shrink", "grow", "insert", "extend", "+=", "*=", "clear"]):
    attempt(b, name, i)
b[0:2] = b"XY"
b[::2] = b"123"
del b[5:5]
b *= 1
print("same size ok", b)
try:
    del b[5:5:2]
except BufferError as e:
    print("empty extended", "BufferError", e)
try:
    b.remove(122)
except ValueError as e:
    print("ValueError first", e)
m.release()
b.pop()
print("released", b)
e = bytearray()
em = memoryview(e)
e.clear()
print("clear empty ok", e)
it = iter(memoryview(b"abc"))
print(next(it))
v = memoryview(b"xyz")
iv = iter(v)
print(next(iv))
v.release()
try:
    print(next(iv))
except ValueError as e2:
    print("ValueError", e2)
