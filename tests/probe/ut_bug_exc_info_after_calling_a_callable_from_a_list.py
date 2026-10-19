# WHAT: A callable taken from a `list[Callable[[], object]]` is called
#   inside a `try` whose handler asks sys.exc_info(). The ownership verifier
#   stops it: "conditionally owned resource from traceback._current_tb ...
#   is used before its union tag proves the payload active". `except
#   Exception as e` instead of sys.exc_info() compiles;
#   `traceback.format_exception(e)` in the handler fails the same way, and
#   so does a callable PARAMETER once a call site passes it one taken from a
#   list. unittest runs each cleanup this way.
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   conditionally owned resource from @_ZN9traceback11_current_tbE result 1 is used before its union tag proves the payload active
# CPython 3.14 expects: caught KeyError / done
import sys
from collections.abc import Callable


def boom() -> object:
    raise KeyError("k")


cleanups: list[Callable[[], object]] = [boom]
try:
    cleanups[0]()
except Exception:
    exc_info = sys.exc_info()
    print("caught", exc_info[0].__name__)
print("done")
