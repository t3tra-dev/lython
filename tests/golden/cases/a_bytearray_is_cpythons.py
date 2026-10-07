# WHAT: bytearray as bytearrayobject.c makes it -- built empty, zeroed, copied,
# from ints and from encoded text; read, written and deleted by index and by
# slice (an extended slice taking exactly as many bytes, and an empty value
# deleting it); grown and shrunk by append, extend, insert, pop, remove, clear,
# + and *; iterated while it changes; its own value as an argument (extend and
# slice assignment copy it first, `b += b` is CPython's BufferError); the
# bytes methods answering with bytearrays; and every refusal in CPython's
# words.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the payload moves as it
# grows and shrinks, and what each operation leaves behind -- the bytes, the
# length, the exception and its message -- is only visible in what the program
# prints after it.

b = bytearray(b"hello")
print(b, len(b), b[0], b[-1], b[1:3], bool(b), bool(bytearray()))
b[0] = 72
b.append(33)
print(b)
b.extend(b" world")
b.extend([33, 63])
print(b, b.count(b"l"), b.find(b"world"), b.startswith(b"Hel"))
print(b.upper(), b.decode(), b.hex()[:6], list(b[:3]))
print(bytearray(3), bytearray([65, 66]), bytearray("é", "utf-8"), bytearray(b"x") + b"y", b"x" + bytearray(b"y"))
print(bytearray(b"ab") * 3, bytearray(b"ab") == b"ab", b"ab" == bytearray(b"ab"), bytearray(b"a") < b"b")
b = bytearray(b"hello")
try:
    print(b[10])
except IndexError as e:
    print("IndexError", e)
for v in [256, -1]:
    try:
        b[0] = v
    except ValueError as e:
        print("ValueError", e)
try:
    b[10] = 1
except IndexError as e:
    print("IndexError", e)
try:
    del b[10]
except IndexError as e:
    print("IndexError", e)
try:
    bytearray().pop()
except IndexError as e:
    print("IndexError", e)
try:
    b.pop(10)
except IndexError as e:
    print("IndexError", e)
try:
    b.remove(122)
except ValueError as e:
    print("ValueError", e)
try:
    b.append(300)
except ValueError as e:
    print("ValueError", e)
try:
    b.insert(0, 256)
except ValueError as e:
    print("ValueError", e)
try:
    print(bytearray(-1))
except ValueError as e:
    print("ValueError", e)
try:
    print(bytearray([1, 256]))
except ValueError as e:
    print("ValueError", e)
print(repr(bytearray(b"a'b\"c\x00\xff")), str(bytearray()))
print(bytearray(b"ab") * 3, 2 * bytearray(b"ab"), bytearray(b"ab") * -1)
print(bytearray(b"abc")[1:], bytearray(b"abc")[::-1], bytearray(b" ab ").strip(), bytearray(b"a,b").split(b","))
print(bytearray(b"-").join([b"a", b"b"]), bytearray.fromhex("6162"), 98 in bytearray(b"ab"), b"b" in bytearray(b"ab"))
print(bytes(bytearray(b"xy")), bytearray(b"abc").find(b"c", 1, 2), bytearray(b"ab") != b"ab")
try:
    print(300 in bytearray(b"ab"))
except ValueError as e:
    print("ValueError", e)
x = bytearray(b"abc")
x[1:2] = b"XYZ"
print(x)
x[::2] = [1, 2, 3]
print(x)
try:
    x[::2] = b"12"
except ValueError as e:
    print("ValueError", e)
del x[::2]
print(x)
x += b"!!"
print(x)
x *= 2
print(x)
x.extend(b"ab")
x.extend([1, 2])
x.extend(bytearray(b"z"))
print(x)
x.reverse()
print(x, x.copy())
x.clear()
print(x, len(x))
y = bytearray(b"aa")
y.extend(y)
print(y)
y[0:1] = y
print(y)
y[::-1] = y
print(y)
z = bytearray(b"ab")
try:
    z += z
except BufferError as e:
    print("BufferError", e)
z *= -1
print(z)
w = bytearray(b"abc")
it = iter(w)
print(next(it))
w.append(100)
print(list(it))
w2 = bytearray(b"abc")
it2 = iter(w2)
print(next(it2))
w2.clear()
print(list(it2))
print(bytearray(b"abc").pop(-1))
try:
    bytearray(b"abc").pop(-4)
except IndexError as e:
    print("IndexError", e)
q = bytearray(b"abc")
q.insert(-10, 120)
q.insert(100, 121)
print(q)
q2 = bytearray(b"abca")
q2.remove(97)
print(q2)
e = bytearray(b"abcd")
e[::2] = b""
print(e)
f = bytearray(b"0123456789")
del f[1:8:3]
print(f)
del f[::-2]
print(f)
g = bytearray(b"abc")
g[5:2] = b"Z"
print(g)
