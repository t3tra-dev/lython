# WHAT: a codec error says what went wrong and where, as CPython's do. A UTF-8
# decode that fails raises UnicodeDecodeError('utf-8', <input>, start, end,
# reason) with the reason and the span CPython's decoder gives -- an invalid
# start byte, an invalid continuation byte after the valid ones (the narrowed
# second-byte ranges of E0/ED/F0/F4 included), or an input that ends inside a
# sequence -- and its message names the byte or the span. Codec errors built
# by the program render their messages the same way (one byte or character
# named, \xhh/\uhhhh/\Uhhhhhhhh escapes, otherwise "start-(end - 1)"), and
# expose encoding/object/start/end/reason.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the reason and the span
# depend on the bytes decoded at run time, and the message is built from the
# exception's arguments when it is raised; only running the decode shows what
# it says.
cases = [b"\xff", b"abc\x80def", b"\xc3", b"\xe0\x80\x80", b"\xed\xa0\x80",
         b"\xe2\x82", b"\xe2\x28\xa1", b"\xf0\x9f\x98", b"\xf0\x9f\x28\x8c",
         b"\xf0\x9f\x98\x28", b"\xf4\x90\x80\x80", b"\xf5\x80", b"\xc0\xaf",
         b"ok\xe2\x82\xacx\xff"]
for data in cases:
    try:
        print(data.decode())
    except UnicodeDecodeError as e:
        print(e)
        print(repr(e), e.start, e.end, e.reason, len(e.object))
try:
    str(b"caf\xc3(", "utf-8")
except UnicodeDecodeError as e:
    print(e.encoding, e.object, e.object[e.start:e.end])
print(str(b"caf\xc3\xa9", "utf-8"), b"good \xf0\x9f\x98\x80 text".decode())
built = [
    UnicodeDecodeError("utf-8", b"ab\xffcd", 2, 3, "invalid start byte"),
    UnicodeDecodeError("ascii", b"abcdef", 1, 4, "ordinal not in range(128)"),
    UnicodeDecodeError("utf-8", b"x", 5, 6, "out of range"),
]
for d in built:
    print(str(d), d.args)
encoded = [
    UnicodeEncodeError("ascii", "caf\xe9", 3, 4, "ordinal not in range(128)"),
    UnicodeEncodeError("ascii", "€\U0001f600", 0, 1, "nope"),
    UnicodeEncodeError("latin-1", "x\U0001f600", 1, 2, "nope"),
    UnicodeEncodeError("ascii", "abc", 0, 3, "nope"),
]
for e in encoded:
    print(str(e), e.encoding, e.object[e.start], e.end)
for t in [UnicodeTranslateError("\xe9", 0, 1, "no mapping"),
          UnicodeTranslateError("abc", 1, 3, "no mapping")]:
    print(str(t), t.object, t.start, t.end, t.reason)
# Arguments built at run time, which a literal's immortal str would not
# show being released twice.
def spelled(stem: str, n: int) -> str:
    return stem + "-" * n
heap = UnicodeDecodeError(spelled("utf", 1) + "8", b"\x00\xff", 1, 2, spelled("bad", 2))
print(heap, heap.encoding, heap.reason, heap.args)
