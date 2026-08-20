# bytes declares Sequence but had neither ordering nor __iter__: `b"a" < b"b"`
# was refused at emit and `for c in b"ab"` at lowering. Must run: both the
# comparison results and the element values (ints, not one-byte bytes) are
# runtime answers.

# Lexicographic over unsigned bytes; a prefix orders first.
print(b"ab" < b"ac", b"ab" <= b"ab", b"b" > b"ab", b"" < b"a")
print(b"abc" >= b"abc", b"abc" > b"abd", b"a" <= b"")

# The compare is unsigned even though C's char is not: 0xff outranks 0x01.
print(b"\xff" > b"\x01", b"\x80" > b"\x7f")

# Ordering reaches the generic helpers, not just the operators.
print(sorted([b"b", b"a", b"ab"]))
print(min(b"zz", b"aa"), max(b"a", b"b"))

# Iteration yields ints.
print([x for x in b"ab"])
for c in b"hi":
    print(c)
print(list(b"AZ"), sum(b"\x01\x02"), max(b"abc"))
print(sorted(b"cab"))

empty: bytes = b""
print([x for x in empty])

# A second pass over the same object sees the whole sequence again.
data = b"xyz"
print(list(data), list(data))
