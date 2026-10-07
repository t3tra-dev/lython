# WHAT: memoryview as memoryobject.c makes it over bytes and bytearray, format
# 'B' -- items read, written and sliced (a slice a view of the same memory,
# reversed and strided ones included, with their strides and contiguity); the
# attributes; equality with bytes, bytearray and other views in either order;
# tobytes, tolist, hex, bytes(view), bytearray(view); hashing a read-only view
# and refusing a writable one; read-only views refusing writes; release() and
# `with` ending the view, and every refusal in CPython's words.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a view reads and writes the
# memory of the object it views, so what it shows -- and what the bytearray
# holds after a write through it -- is only visible at run time.

b = bytearray(b"hello")
m = memoryview(b)
print(len(m), m[1], m[-1], m[1:3].tobytes(), m[::-1].tobytes(), m[::2].tolist())
print(m.readonly, m.format, m.itemsize, m.nbytes, m.ndim, m.shape, m.strides, m.contiguous, m.c_contiguous, m.f_contiguous)
print(m[::-1].strides, m[::-2].shape, m[::-1].contiguous)
print(m == b"hello", m == bytearray(b"hello"), m == memoryview(b"hello"), m == b"x", m != b"hello")
print(b"hello" == m, bytearray(b"hello") == m, b"x" != m)
try:
    print(m[10])
except IndexError as e:
    print("IndexError", e)
try:
    b.append(1)
except BufferError as e:
    print("BufferError", e)
m[0] = 72
print(b)
try:
    m[0] = 256
except ValueError as e:
    print("ValueError", e)
m[0:2] = b"HE"
print(b)
try:
    m[0:2] = b"H"
except ValueError as e:
    print("ValueError", e)
print(m.hex(), bytes(m), list(m), bytearray(m))
try:
    print(hash(m))
except ValueError as e:
    print("ValueError", e)
r = memoryview(b"abc")
try:
    r[0] = 1
except TypeError as e:
    print("TypeError", e)
print(hash(r) == hash(b"abc"), r.readonly, r.toreadonly().readonly, m.toreadonly().readonly)
m.release()
try:
    print(m[0])
except ValueError as e:
    print("ValueError", e)
try:
    print(len(m))
except ValueError as e:
    print("ValueError", e)
b.append(33)
print(b)
with memoryview(b) as w:
    print(w[0])
try:
    print(w.tobytes())
except ValueError as e:
    print("ValueError", e)
v = memoryview(b)
s = v[1:3]
v.release()
try:
    b.append(1)
except BufferError as e:
    print("BufferError", e)
s.release()
b.append(2)
print(len(b))
mm = memoryview(memoryview(b"xyz"))
print(mm.tobytes(), [x for x in memoryview(b"ab")], 98 in memoryview(b"ab"), 300 in memoryview(b"ab"), memoryview(b"abc")[::-1].tolist())
print(repr(memoryview(b"x"))[:13])
