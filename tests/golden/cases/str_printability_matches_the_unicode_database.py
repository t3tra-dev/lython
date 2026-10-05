# WHAT: str.isprintable over every code point, and the width repr gives a
# sampled seventh of them, against CPython 3.14's database: the count and an
# ordered digest of the printable code points, and one string whose repr
# escapes each kind of unprintable character (Zs, Cf, Cs, Cc) and passes the
# printable ones through.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: UnicodeTablesTest checks
# the printable bounds against the category table in C++. What it cannot run
# is the search the runtime does over them (_ucd.mlir, __ly_ucd_is_printable),
# which only answers when the compiled program asks it.
count = 0
digest = 0
for cp in range(0x110000):
    if chr(cp).isprintable():
        count += 1
        digest = (digest * 31 + cp) % 1000000007
print(count, digest)
widths = 0
for cp in range(0, 0x110000, 7):
    widths += len(repr(chr(cp)))
print(widths)
print(repr("a\u00a0b\u3042\U0001f600\u200b" + chr(0xD800) + "\U000e0001\x7f"))
