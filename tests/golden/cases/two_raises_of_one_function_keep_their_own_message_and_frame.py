# Two `raise cls("...")` statements in one function each raise their own
# exception from their own line. A cleanup pass merged the two raise blocks --
# identical but for their locations -- into one: the second raise pushed the
# first one's traceback frame, and its exception, handed over on an edge
# rewired ahead of where it was lowered, lost its message
# (`AssertionError()`). unittest's `fail` is written this way.
# Why run it: the message and the frame are what the run raises.
import sys
import traceback


def check(cls: type[BaseException], which: bool) -> None:
    if which:
        raise cls("first")
    raise cls("second")


for which in [True, False]:
    try:
        check(AssertionError, which)
    except AssertionError as e:
        lines = [entry.split(", line ")[1].split("\n")[0]
                 for entry in traceback.format_tb(sys.exc_info()[2])]
        print(repr(e), lines)
