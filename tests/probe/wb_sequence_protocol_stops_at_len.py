# The `__len__`/`__getitem__` sequence protocol stops at `__len__` here and at
# the first IndexError in CPython, and a class whose two disagree gets two
# different answers with no diagnostic:
#
#     __len__ says 2, __getitem__ raises at 4
#     CPython: [0, 1, 2, 3]      lyc: [0, 1]
#
# ⭐ WHY IT IS NOT A BUG FOR ANY CLASS ANYONE WRITES, and why it is recorded
# anyway: the two agree for every well-formed sequence, which is the case the
# rewrite was built for (`emitSequenceProtocolFor`, EmitterLoops.cpp, turns
# `for v in seq` into an index loop bounded by `len(seq)`). It is on this list
# because the divergence is SILENT -- the project's own rule is that an
# unsupported shape is refused at the earliest static boundary, and this one
# answers instead.
#
# ⛔ REFUSING IS NOT THE REPAIR EITHER: a class cannot prove its `__len__`
# agrees with its `__getitem__`, so the refusal would land on every correct
# sequence class.
#
# THE SHAPE OF THE REPAIR: the loop CPython runs is
#
#     i = 0
#     while True:
#         try:
#             v = s[i]
#         except IndexError:
#             break
#         ...
#         i += 1
#
# and the synthesizer has no `try` builder (`AstSynth.h` stops at while/for/if).
# Adding one is the whole cost; the rest is the same rewrite with a different
# bound. `__iter__` classes are unaffected -- they carry their own end.
class Odd:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index >= 4:
            raise IndexError("end")
        return index


print(list(Odd()), len(Odd()))
