# WHAT: The typical ParamSpec decorator: an inner `def inner(*args: P.args,
#   **kwargs: P.kwargs)` returned as the decorated function. A forwarder
#   CALLED with its arguments is compiled per call shape; one returned as a
#   VALUE would need the inner def specialized at the decorated function's
#   signature.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a `**` mapping whose keys are known only at run time can be passed here only to the **kwargs of a function or method this program defines
# CPython 3.14 expects: calling / calling / 3 7
from collections.abc import Callable


def logged[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        print("calling")
        return f(*args, **kwargs)
    return inner


@logged
def add(a: int, b: int = 1) -> int:
    return a + b


print(add(2), add(2, b=5))
