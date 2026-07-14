# open() with a str-literal 'b' mode statically selects the binary arm and
# returns the raw FileIO (CPython returns a Buffered* wrapper; the Lib/io.py
# port's wrappers delegate to FileIO 1:1).
f = open("open_binary_case.tmp", "wb")
print(f.write(b"binary\x00mode"))
f.close()
g = open("open_binary_case.tmp", "rb")
print(g.read())
print(g.seek(0))
print(g.read(6))
print(g.readable())
print(g.writable())
g.close()
t = open("open_binary_case.tmp")
print(t.readable())
t.close()
