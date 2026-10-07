# WHAT: a size no allocation can meet raises what CPython raises, and the
# program goes on: a list, tuple, str or bytes repeated past memory or past the
# word, `bytes(n)`, str padding, an int shifted past memory or past the digits
# an int may have, a repeat count past the word of either sign -- MemoryError
# where the size is only too big, OverflowError where it does not fit (an int
# of more digits than CPython's MAX_LONG_DIGITS, at the same boundary), caught in
# the frame that asked and in a caller -- and a count that wraps is never a
# small block written past (`[0] * (2**64 + 1)` is not a one-element list); an
# empty sequence repeated any number of times is empty at once. Format widths
# and precisions too: a width past memory is MemoryError, one past the word
# "Too many decimal digits", a float precision past a C int "precision too
# big", and `expandtabs` takes its tab size as a C int.
# Where Lython's message for an int argument past the word is not CPython's
# (`'x' * (2**64 + 1)`, `bytes(2**64)`, `str.center`), only the type is shown.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the sizes are values the
# program computes; the allocator or a size check raises while it runs, and
# only a run shows the exception caught and the program going on with its
# memory intact.


def pow2(e: int) -> int:
    return 2 ** e


def repeat_in_place(n: int) -> int:
    xs = [0, 1]
    xs *= n
    return len(xs)


def grow(n: int) -> list[int]:
    return [0] * n


try:
    print('[0] * 2**62', "ok", len([0] * pow2(62)))
except MemoryError as e:
    print('[0] * 2**62', "MemoryError", e)
except OverflowError as e:
    print('[0] * 2**62', "OverflowError", e)
try:
    print('[0, 1] * 2**62', "ok", len([0, 1] * pow2(62)))
except MemoryError as e:
    print('[0, 1] * 2**62', "MemoryError", e)
except OverflowError as e:
    print('[0, 1] * 2**62', "OverflowError", e)
try:
    print('(0,) * 2**62', "ok", len((0,) * pow2(62)))
except MemoryError as e:
    print('(0,) * 2**62', "MemoryError", e)
except OverflowError as e:
    print('(0,) * 2**62', "OverflowError", e)
try:
    print('[0, 0, 0] * 2**62', "ok", len([0, 0, 0] * pow2(62)))
except MemoryError as e:
    print('[0, 0, 0] * 2**62', "MemoryError", e)
except OverflowError as e:
    print('[0, 0, 0] * 2**62', "OverflowError", e)
try:
    print('[0] * (2**64 + 1)', "ok", len([0] * (pow2(64) + 1)))
except MemoryError as e:
    print('[0] * (2**64 + 1)', "MemoryError", e)
except OverflowError as e:
    print('[0] * (2**64 + 1)', "OverflowError", e)
try:
    print('(0,) * (2**64 + 1)', "ok", len((0,) * (pow2(64) + 1)))
except MemoryError as e:
    print('(0,) * (2**64 + 1)', "MemoryError", e)
except OverflowError as e:
    print('(0,) * (2**64 + 1)', "OverflowError", e)
try:
    print('[0] * 2**63', "ok", len([0] * pow2(63)))
except MemoryError as e:
    print('[0] * 2**63', "MemoryError", e)
except OverflowError as e:
    print('[0] * 2**63', "OverflowError", e)
try:
    print('[0] * -2**70', "ok", len([0] * -pow2(70)))
except MemoryError as e:
    print('[0] * -2**70', "MemoryError", e)
except OverflowError as e:
    print('[0] * -2**70', "OverflowError", e)
try:
    print('[] * 2**62', "ok", len([] * pow2(62)))
except MemoryError as e:
    print('[] * 2**62', "MemoryError", e)
except OverflowError as e:
    print('[] * 2**62', "OverflowError", e)
try:
    print('() * 2**62', "ok", len(() * pow2(62)))
except MemoryError as e:
    print('() * 2**62', "MemoryError", e)
except OverflowError as e:
    print('() * 2**62', "OverflowError", e)
try:
    print("'' * 2**62", "ok", len('' * pow2(62)))
except MemoryError as e:
    print("'' * 2**62", "MemoryError", e)
except OverflowError as e:
    print("'' * 2**62", "OverflowError", e)
try:
    print("b'' * 2**62", "ok", len(b'' * pow2(62)))
except MemoryError as e:
    print("b'' * 2**62", "MemoryError", e)
except OverflowError as e:
    print("b'' * 2**62", "OverflowError", e)
try:
    print("'ab' * 2**62", "ok", len('ab' * pow2(62)))
except MemoryError as e:
    print("'ab' * 2**62", "MemoryError", e)
except OverflowError as e:
    print("'ab' * 2**62", "OverflowError", e)
try:
    print("b'ab' * 2**62", "ok", len(b'ab' * pow2(62)))
except MemoryError as e:
    print("b'ab' * 2**62", "MemoryError", e)
except OverflowError as e:
    print("b'ab' * 2**62", "OverflowError", e)
try:
    print("'x' * (2**63 - 1)", "ok", len('x' * (pow2(63) - 1)))
except MemoryError as e:
    print("'x' * (2**63 - 1)", "MemoryError", e)
except OverflowError as e:
    print("'x' * (2**63 - 1)", "OverflowError", e)
try:
    print("'x' * (2**64 + 1)", "ok", len('x' * (pow2(64) + 1)))
except MemoryError as e:
    print("'x' * (2**64 + 1)", "MemoryError")
except OverflowError as e:
    print("'x' * (2**64 + 1)", "OverflowError")
try:
    print("b'x' * (2**64 + 1)", "ok", len(b'x' * (pow2(64) + 1)))
except MemoryError as e:
    print("b'x' * (2**64 + 1)", "MemoryError")
except OverflowError as e:
    print("b'x' * (2**64 + 1)", "OverflowError")
try:
    print('bytes(2**62)', "ok", len(bytes(pow2(62))))
except MemoryError as e:
    print('bytes(2**62)', "MemoryError", e)
except OverflowError as e:
    print('bytes(2**62)', "OverflowError", e)
try:
    print('bytes(2**50)', "ok", len(bytes(pow2(50))))
except MemoryError as e:
    print('bytes(2**50)', "MemoryError", e)
except OverflowError as e:
    print('bytes(2**50)', "OverflowError", e)
try:
    print('bytes(2**64)', "ok", len(bytes(pow2(64))))
except MemoryError as e:
    print('bytes(2**64)', "MemoryError")
except OverflowError as e:
    print('bytes(2**64)', "OverflowError")
try:
    print("'x'.ljust(2**62)", "ok", len('x'.ljust(pow2(62))))
except MemoryError as e:
    print("'x'.ljust(2**62)", "MemoryError", e)
except OverflowError as e:
    print("'x'.ljust(2**62)", "OverflowError", e)
try:
    print("'x'.zfill(2**62)", "ok", len('x'.zfill(pow2(62))))
except MemoryError as e:
    print("'x'.zfill(2**62)", "MemoryError", e)
except OverflowError as e:
    print("'x'.zfill(2**62)", "OverflowError", e)
try:
    print("'x'.center(2**64)", "ok", len('x'.center(pow2(64))))
except MemoryError as e:
    print("'x'.center(2**64)", "MemoryError")
except OverflowError as e:
    print("'x'.center(2**64)", "OverflowError")
try:
    print('1 << 2**62', "ok", (1 << pow2(62)).bit_length())
except MemoryError as e:
    print('1 << 2**62', "MemoryError", e)
except OverflowError as e:
    print('1 << 2**62', "OverflowError", e)
try:
    print('1 << (2**63 - 1)', "ok", (1 << (pow2(63) - 1)).bit_length())
except MemoryError as e:
    print('1 << (2**63 - 1)', "MemoryError", e)
except OverflowError as e:
    print('1 << (2**63 - 1)', "OverflowError", e)
try:
    print('1 << 2**64', "ok", (1 << pow2(64)).bit_length())
except MemoryError as e:
    print('1 << 2**64', "MemoryError", e)
except OverflowError as e:
    print('1 << 2**64', "OverflowError", e)
try:
    print('1 << (2**63 - 38)', "ok", (1 << (pow2(63) - 38)).bit_length())
except MemoryError as e:
    print('1 << (2**63 - 38)', "MemoryError", e)
except OverflowError as e:
    print('1 << (2**63 - 38)', "OverflowError", e)
try:
    print('1 << (2**63 - 37)', "ok", (1 << (pow2(63) - 37)).bit_length())
except MemoryError as e:
    print('1 << (2**63 - 37)', "MemoryError", e)
except OverflowError as e:
    print('1 << (2**63 - 37)', "OverflowError", e)
try:
    print('0 << 2**100', "ok", 0 << pow2(100))
except MemoryError as e:
    print('0 << 2**100', "MemoryError", e)
except OverflowError as e:
    print('0 << 2**100', "OverflowError", e)
try:
    print('-1 >> 2**100', "ok", -1 >> pow2(100))
except MemoryError as e:
    print('-1 >> 2**100', "MemoryError", e)
except OverflowError as e:
    print('-1 >> 2**100', "OverflowError", e)
try:
    print('xs *= 2**62', "ok", repeat_in_place(pow2(62)))
except MemoryError as e:
    print('xs *= 2**62', "MemoryError", e)
except OverflowError as e:
    print('xs *= 2**62', "OverflowError", e)
try:
    print('a list made in a callee', "ok", len(grow(pow2(62))))
except MemoryError as e:
    print('a list made in a callee', "MemoryError", e)
except OverflowError as e:
    print('a list made in a callee', "OverflowError", e)
try:
    print("f'{1:{2**62}}'", "ok", len(f'{1:{pow2(62)}}'))
except MemoryError as e:
    print("f'{1:{2**62}}'", "MemoryError", e)
except OverflowError as e:
    print("f'{1:{2**62}}'", "OverflowError", e)
except ValueError as e:
    print("f'{1:{2**62}}'", "ValueError", e)
try:
    print("format('x', 2**62)", "ok", len(format('x', str(pow2(62)))))
except MemoryError as e:
    print("format('x', 2**62)", "MemoryError", e)
except OverflowError as e:
    print("format('x', 2**62)", "OverflowError", e)
except ValueError as e:
    print("format('x', 2**62)", "ValueError", e)
try:
    print('format(1.0, 2**62)', "ok", len(format(1.0, str(pow2(62)))))
except MemoryError as e:
    print('format(1.0, 2**62)', "MemoryError", e)
except OverflowError as e:
    print('format(1.0, 2**62)', "OverflowError", e)
except ValueError as e:
    print('format(1.0, 2**62)', "ValueError", e)
try:
    print('format(1, 2**63)', "ok", len(format(1, str(pow2(63)))))
except MemoryError as e:
    print('format(1, 2**63)', "MemoryError", e)
except OverflowError as e:
    print('format(1, 2**63)', "OverflowError", e)
except ValueError as e:
    print('format(1, 2**63)', "ValueError", e)
try:
    print("format(1.0, '.2**62f')", "ok", len(format(1.0, '.' + str(pow2(62)) + 'f')))
except MemoryError as e:
    print("format(1.0, '.2**62f')", "MemoryError", e)
except OverflowError as e:
    print("format(1.0, '.2**62f')", "OverflowError", e)
except ValueError as e:
    print("format(1.0, '.2**62f')", "ValueError", e)
try:
    print("format(1.0, '.2**31f')", "ok", len(format(1.0, '.' + str(pow2(31)) + 'f')))
except MemoryError as e:
    print("format(1.0, '.2**31f')", "MemoryError", e)
except OverflowError as e:
    print("format(1.0, '.2**31f')", "OverflowError", e)
except ValueError as e:
    print("format(1.0, '.2**31f')", "ValueError", e)
try:
    print("'%4611686018427387904d' % 1", "ok", len('%4611686018427387904d' % 1))
except MemoryError as e:
    print("'%4611686018427387904d' % 1", "MemoryError", e)
except OverflowError as e:
    print("'%4611686018427387904d' % 1", "OverflowError", e)
except ValueError as e:
    print("'%4611686018427387904d' % 1", "ValueError", e)
try:
    print("'%.4611686018427387904f' % 1.0", "ok", len('%.4611686018427387904f' % 1.0))
except MemoryError as e:
    print("'%.4611686018427387904f' % 1.0", "MemoryError", e)
except OverflowError as e:
    print("'%.4611686018427387904f' % 1.0", "OverflowError", e)
except ValueError as e:
    print("'%.4611686018427387904f' % 1.0", "ValueError", e)
try:
    print('expandtabs(2**31)', "ok", len('a\tb'.expandtabs(pow2(31))))
except MemoryError as e:
    print('expandtabs(2**31)', "MemoryError", e)
except OverflowError as e:
    print('expandtabs(2**31)', "OverflowError", e)
except ValueError as e:
    print('expandtabs(2**31)', "ValueError", e)
try:
    print('expandtabs(-2**50)', "ok", len('a\tb'.expandtabs(-pow2(50))))
except MemoryError as e:
    print('expandtabs(-2**50)', "MemoryError", e)
except OverflowError as e:
    print('expandtabs(-2**50)', "OverflowError", e)
except ValueError as e:
    print('expandtabs(-2**50)', "ValueError", e)
print("after", [1, 2, 3] * 2, "ab" * 3, len(bytes(4)), 1 << 70, format(7, "05d"), format(2.5, ".3f"), "a\tb".expandtabs(4))
print("whole digits", [((1 << 100) << k).bit_length() for k in (30, 60, 90)], (-5 << 60) >> 60)
