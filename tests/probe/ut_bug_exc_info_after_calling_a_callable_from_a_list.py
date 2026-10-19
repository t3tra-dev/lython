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
# FIXED: 2026-10-20 -- boxing `Tb | None` into object reads each member's class
#   word before the tag picks; a load of the inactive member's placeholder is
#   not a use (cases/an_optional_instance_boxed_into_object).
# CLASSIFICATION @ lang-gaps 326fa275 + the fix: 1 正しい
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
