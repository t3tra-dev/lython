# What this pins: the ASCII half of bytes' method table -- eighteen entries the
# manifest did not have, found by asking CPython which names `dir(b"")` has that
# this compiler rejects.
#
#     print(b"hello".upper())
#     # static type !py.contract<"builtins.bytes"> does not provide manifest
#     # method 'upper'
#
# Every one of them is a byte-wise walk, which is also what CPython does: a
# bytes object has no encoding, so `.upper()` maps a..z and leaves every other
# byte -- including a UTF-8 continuation byte -- exactly where it is. The five
# case maps share one loop because capitalize and title need the POSITION, and
# the seven predicates share another because islower/isupper need "was there a
# cased byte at all" alongside "did every byte pass".
#
# Why this must run: these are answers, not shapes. The empty bytes is the case
# that separates the predicates -- False for six of them and True for isascii --
# and a high byte is what separates a byte-wise map from a text one.
#
# ⛔ istitle, index, rindex, rfind, ljust, rjust, center, zfill, expandtabs,
# partition, rpartition, rsplit, splitlines, maketrans and translate are still
# missing. They are not harder, only more; this is the batch that the case maps
# and the character classes pay for.
data = b"Hello World"
print(data.upper(), data.lower(), data.swapcase())
print(data.capitalize(), data.title(), b"hello world".title())
print(b"a1b2".upper(), b"\xc3\xa9abc".upper(), b"".upper())

print(b"  xx  ".lstrip(), b"  xx  ".rstrip(), b"  xx  ".strip())
print(b"aabxaa".lstrip(b"a"), b"aabxaa".rstrip(b"a"), b"aabxaa".strip(b"a"))

print(b"prefix_body".removeprefix(b"prefix_"), b"body_suffix".removesuffix(b"_suffix"))
print(b"abc".removeprefix(b"z"), b"abc".removesuffix(b"z"))
print(b"a".removeprefix(b"abc"), b"a".removesuffix(b"abc"), b"abc".removeprefix(b""))

for probe in [b"abc", b"ABC", b"aBc", b"a1", b"123", b"  ", b"", b"\xff", b"a b"]:
    print(
        probe.isalpha(),
        probe.isdigit(),
        probe.isalnum(),
        probe.isspace(),
        probe.isascii(),
        probe.islower(),
        probe.isupper(),
    )

n = 0
i = 0
while i < 100:
    n += len(b"abc".upper()) + len(b" x ".strip())
    i += 1
print("loop", n)
