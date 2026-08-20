# What this pins: `in` over bytes, both spellings CPython takes.
#
#     print(b"a" in b"abc")
#     # builtins.bytes.__contains__ is declared by the standard-library
#     # contract but has no runtime implementation
#
# bytes inherits __contains__ from its Sequence base, so the operator
# type-checked and then had nothing to call. CPython accepts a bytes (a
# subsequence test) and an int (a byte-value test) through the same operator,
# so both are implemented.
#
# Why this must run: the answers are values, and two of them are the ones a
# reimplementation gets wrong -- the empty bytes is IN every bytes, and an int
# outside 0..255 RAISES rather than answering False. CPython's own message is
# pinned with it.
#
# ⛔ The bytes form is `find(...) >= 0` rather than a second scan, so the empty
# case comes from the same code that already answers it for find() and the two
# cannot drift apart.
data = b"abcabc"

print(b"a" in data, b"bc" in data, b"abcabc" in data, b"" in data)
print(b"z" in data, b"acb" in data, b"abcabcabc" in data)
print(97 in data, 99 in data, 122 in data, 0 in data)
print(b"a" not in data, b"z" not in data, 97 not in data, 122 not in data)

for bad in [256, -1, 1000]:
    try:
        print(bad in data)
    except ValueError as e:
        print("range", e)

hits = 0
i = 0
while i < 200:
    if b"bc" in data:
        hits += 1
    if (i % 256) in data:
        hits += 1
    i += 1
print("hits", hits)
