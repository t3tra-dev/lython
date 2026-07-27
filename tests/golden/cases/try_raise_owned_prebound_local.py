# An owned local bound BEFORE the try, handed to `raise` inside it, and not read
# by the handler. The concat that builds the message is the local's last use, so
# its release lands there -- ahead of the guarded LyValueError_New / _Init /
# _Raise call sites, every one of which unwinds into a handler whose entry
# releases the same group. Refused before the fix ("owned resource from
# @LyUnicode_Concat result 0 is released or transferred more than once on one CFG
# path"), which is a false rejection: CPython runs it.
#
# Why this needs execution rather than a DriverTests success assertion: the fix
# MOVES a release, so compiling proves only that the arithmetic balances. Whether
# it frees the right object at the right time is visible in the message the
# handler reads back out of the exception.
def wrap(k: int) -> str:
    a = "x" + str(k)
    b = "y" + str(k)
    try:
        raise ValueError(a + b)
    except ValueError as ex:
        return str(ex)[-3:]


def wrap_no_bind(k: int) -> str:
    a = "p" + str(k)
    b = "q" + str(k)
    try:
        raise ValueError(a + b)
    except ValueError:
        return "handled"


print(wrap(3))
print(wrap(41))
print(wrap_no_bind(3))
