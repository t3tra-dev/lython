# WHAT: A builtin function handed to a `Callable[P, object]` parameter whose
#   `*args: P.args` the call fills -- `self.addCleanup(os.remove, path)` in a
#   test, here `later(print, "x", "y")`. A builtin passed where a callable of
#   FIXED parameters is expected becomes a forwarder typed by that callable;
#   through a ParamSpec the callable is the builtin's own signature, `*args`
#   included, and the forwarder cannot take a `*args` yet. Refused in the
#   lowering: "cannot adapt builtins.function to runtime input 0".
#   A bound method of a builtin instance (`later(log.append, "x")`) works:
#   its bound signature has no `*args`. The same refusal reaches a builtin
#   in an arm of a conditional expression handed to a callable parameter
#   (`apply(print if c else log.append, "z")`): the expectation does not
#   reach the arms, so the builtin is not wrapped there.
# KIND: 未対応 (unsupported) -- loud, from the lowering
# FOUND: 2026-10-20, making builtin functions Callable values for the
#   unittest port.
# CLASSIFICATION: loud 拒否 (診断)
#   cannot adapt builtins.function to runtime input 0 of
#   builtins.function._ZN8__main__5laterE
# CPython 3.14 expects: x y
from collections.abc import Callable


def later[**P](f: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> Callable[[], object]:
    return lambda: f(*args, **kwargs)


cb = later(print, "x", "y")
cb()
