# WHAT: `sys.exc_info()` in a helper called FROM an except block. It is
#   answered only lexically inside the handler; CPython answers it for
#   whatever handler is active on the stack.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- outside an except block it is typeshed's optional triple,
#   answered at run time; the program narrows it, as typeshed requires (the
#   unnarrowed `etype.__name__` it was written with is refused).
# CLASSIFICATION @ lang-gaps e3a74bc1 + the fix: 1 正しい
# CPython 3.14 expects: KeyError: 'k'
import sys


def report() -> str:
    etype, value, tb = sys.exc_info()
    if etype is None or value is None:
        return "nothing"
    return etype.__name__ + ": " + str(value)


try:
    raise KeyError("k")
except KeyError:
    print(report())
