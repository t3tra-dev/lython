# WHAT: `sys.exc_info()` in a function called from an except block -- and
# from inside a nested handler, after it completes, and with no handler
# running at all -- narrowed before its members are used, as typeshed's
# optional triple has to be.
#
# WHY THIS IS RUN: which exception is being handled is a fact about the run,
# not the text. The call was refused outside an except block written around
# it, so unittest's helper that reads its caller's exception could not be
# written.
import sys


def report() -> str:
    etype, value, tb = sys.exc_info()
    if etype is None or value is None:
        return "nothing"
    return etype.__name__ + ": " + str(value)


def nested() -> str:
    try:
        raise ValueError("inner")
    except ValueError:
        inner = report()
    return inner + " / then " + report()


print(report())
try:
    raise KeyError("k")
except KeyError:
    print(report())
    print(nested())
    print(report())
print(report())
