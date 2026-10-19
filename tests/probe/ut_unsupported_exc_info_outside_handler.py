# WHAT: `sys.exc_info()` in a helper called FROM an except block. It is
#   answered only lexically inside the handler; CPython answers it for
#   whatever handler is active on the stack.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   sys.exc_info() is answered inside an except block, where the exception it reports is the one that block handles: elsewhere it is the handler active on the stack at run time
# CPython 3.14 expects: KeyError: 'k'
import sys


def report() -> str:
    etype, value, tb = sys.exc_info()
    return etype.__name__ + ": " + str(value)


try:
    raise KeyError("k")
except KeyError:
    print(report())
