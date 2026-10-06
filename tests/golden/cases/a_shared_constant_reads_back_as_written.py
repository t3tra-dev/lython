# WHAT: literals that are one static object shared by every evaluation -- str
# literals of every code-unit width, the empty str, every one-code-point str
# below 256 (indexing, iteration, chr), bytes literals, the empty tuple and
# tuples of small ints and None -- read back as written through every kind of
# reader, in loops that would have allocated each one.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the objects are laid out
# by the compiler as bytes in read-only data (header words, shape word,
# relocated addresses), and only the runtime's own readers decoding them shows
# the layout matches what those readers expect.


def words() -> list[str]:
    out: list[str] = []
    for i in range(3):
        out.append("ascii")
        out.append("é latin")
        out.append("日本")
        out.append("😀 wide")
        out.append("")
    return out


ws = words()
print(ws[:5], [len(w) for w in ws[:5]], ws[1].upper(), ws[3][0] == "😀")
print("".join(ws[:4]), ws[2] + ws[4] + "x", ws[0] == "ascii", hash(ws[0]) == hash("as" + "cii"))
s = "héllo"
print([c for c in s], s[1], ord(s[1]), chr(255), chr(256), [chr(c) for c in range(250, 258)])
acc = ""
for c in "abc":
    acc += c
print(acc, acc == "abc", "" + "" == "", len(""), "abc"[1:1] == "")
bs = [b"bytes" for _ in range(3)]
print(bs, bs[0][1], len(bs[2]), bs[0] + b"!", b"" + b"", bs[1].upper())
ts = [(1, -2) for _ in range(3)]
nn = [(None, 0, -(2 ** 30)) for _ in range(2)]
print(ts, nn, ts[0][1], len(nn[0]), ts[0] == (1, -2), hash(ts[1]) == hash((1, -2)))
a, b = ts[2]
print(a + b, [() for _ in range(2)], len(()), () + (1,))
