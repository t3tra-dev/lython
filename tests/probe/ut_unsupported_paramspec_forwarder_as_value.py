# WHAT: A ParamSpec forwarder bound to a name and called through it: it has
#   no value of its own, only a body per call shape.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   reference to generic function 'call' requires a call or an annotated Callable context to determine its type arguments
# CPython 3.14 expects: 3
from collections.abc import Callable


def call[**P, R](f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return f(*args, **kwargs)


def add(a: int, b: int) -> int:
    return a + b


g = call
print(g(add, 1, 2))
