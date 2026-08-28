# WHAT: a function with a non-None result annotation that never reaches its
# end -- every path raises. It has no fallthrough to give a value to, and the
# exception it raises has to arrive at the caller.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what is checked is that the
# raise CROSSES the boundary the missing return would have been at. The failure
# it replaces was a return the ABI could not expand ("expected 2 physical
# values, but lowering produced 0"), and any repair that fabricates a value
# instead compiles just as well and hands the caller a made-up string.
import sys


def reraise() -> str:
    try:
        raise ValueError("inner")
    except ValueError:
        raise


def chained() -> str:
    try:
        raise ValueError("first")
    except ValueError as e:
        raise RuntimeError("second") from e


def always(n: int) -> int:
    if n > 0:
        raise ValueError("positive")
    raise KeyError("other")


def guarded(n: int) -> str:
    try:
        if n > 0:
            return "positive"
        raise ValueError("not positive")
    except ValueError as e:
        return "caught " + str(e)


try:
    reraise()
except ValueError as e:
    sys.stdout.write("reraise: " + str(e) + "\n")

try:
    chained()
except RuntimeError as e:
    sys.stdout.write("chained: " + str(e) + "\n")

try:
    always(1)
except ValueError as e:
    sys.stdout.write("always+: " + str(e) + "\n")

try:
    always(0)
except KeyError as e:
    sys.stdout.write("always0: " + str(e) + "\n")

sys.stdout.write(guarded(1) + "\n")
sys.stdout.write(guarded(-1) + "\n")
