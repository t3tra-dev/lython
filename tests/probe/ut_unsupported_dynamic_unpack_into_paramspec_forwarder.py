# WHAT: `call(add, *xs)` into a ParamSpec forwarder: the forwarder is
#   compiled per call shape, and a sequence of unknown length has no shape.
# KIND: 未対応 (unsupported) -- by design for now
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a starred sequence of unknown length or a `**` mapping of unknown keys cannot fill `*args: P.args` or `**kwargs: P.kwargs`, whose members are parameters of their own in the body each call is compiled to: spell the argume
# CPython 3.14 expects: 3
from collections.abc import Callable


def call[**P, R](f: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return f(*args, **kwargs)


def add(a: int, b: int) -> int:
    return a + b


xs = [1, 2]
print(call(add, *xs))
