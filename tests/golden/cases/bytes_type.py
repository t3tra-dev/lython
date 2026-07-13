from io import BytesIO

# bytes literals, indexing (byte values, negative wrap), concatenation,
# equality, repr escaping (bytesobject.c semantics), encode/decode.
data = b"hello"
print(len(data))
print(data[0])
print(data[-1])
print(data)
joined = data + b" world\x00\t!"
print(joined)
print(joined == b"hello world\x00\t!")
print(joined != data)
# Truthiness dispatches through the manifest __bool__ (bool(x) as a class
# call is not supported yet for any receiver).
if b"":
    print("wrong: empty bytes is truthy")
else:
    print("empty falsy")
if data:
    print("non-empty truthy")
print(b"text".decode())
encoded = "round trip".encode()
print(encoded)
print(encoded.decode())

try:
    data[99]
except IndexError:
    print("index caught")

# BytesIO (the native _io implementation; read() consumes from the stream
# position, getvalue() does not move it -- bytesio.c semantics).
bio = BytesIO()
print(bio.write(b"raw "))
bio.write("payload".encode())
value = bio.getvalue()
print(value)
print(value.decode())
print(bio.read() == b"")
print(bio.getvalue())
print(bio.readable())
bio.close()
try:
    bio.write(b"x")
except ValueError:
    print("closed caught")
