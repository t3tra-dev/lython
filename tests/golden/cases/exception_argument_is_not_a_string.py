# What this pins: an exception constructed with ONE non-str argument.
#
#     raise ValueError(42)
#     # cannot adapt builtins.int to runtime input 3 of
#     # builtins.ValueError.__init__
#
# The refusal was asymmetric in the direction that gives it away: `ValueError(
# "x", 42)` -- strictly more work -- already compiled, because two arguments
# take the payload block while one goes into the message LANE, which is a
# unicode. So the one-argument case now takes the same block, and .args reads
# the value back as an int rather than as its text.
#
# Why this must run: str(e) is RENDERED from the block, and one argument
# renders as str(arg) where two render as the "(a, b)" tuple -- CPython's
# BaseException.__str__, which no compile-time check can show. `int(str(e))`
# decodes it: a message that merely looked right would not survive being read
# back as a number and summed.
#
# ⛔ KeyError.__str__ IS repr(args[0]), and it is INHERITED, so the block
# renderer asks the class taxonomy rather than comparing one class id. Point
# below has both a __str__ and a __repr__ that differ, which is the only way to
# see that KeyError takes the repr and ValueError takes the str.
#
# ⛔ SystemExit is still refused with a non-str argument. Its exit status is
# recorded out of band and the runner reads an empty message as "use that
# status", so an int argument that reached the block would print its code and
# exit 1 where CPython exits WITH it. sys.exit(n) is the spelling that works.
class Point:
    def __init__(self, x: int) -> None:
        self.x = x

    def __str__(self) -> str:
        return "P(" + str(self.x) + ")"

    def __repr__(self) -> str:
        return "Point(" + str(self.x) + ")"


class Wrapped(ValueError):
    pass


try:
    raise ValueError(42)
except ValueError as e:
    print(str(e), e.args, len(e.args))

try:
    raise IndexError(7)
except IndexError as e:
    print(str(e), e.args)

try:
    raise RuntimeError(1.5)
except RuntimeError as e:
    print(str(e), e.args)

try:
    raise ValueError([1, 2])
except ValueError as e:
    print(str(e), e.args)

try:
    raise ValueError((1, 2))
except ValueError as e:
    print(str(e), len(e.args))

try:
    raise Wrapped(9)
except ValueError as e:
    print(str(e), e.args, type(e).__name__)

try:
    raise KeyError(Point(3))
except KeyError as e:
    print(str(e))

try:
    raise ValueError(Point(3))
except ValueError as e:
    print(str(e))

try:
    raise ValueError("x", 42)
except ValueError as e:
    print(str(e), e.args)

total = 0
i = 0
while i < 50:
    try:
        raise IndexError(i * 3)
    except IndexError as e:
        total += int(str(e))
    i += 1
print(total)
