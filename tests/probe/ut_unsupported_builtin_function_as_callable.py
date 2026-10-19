# WHAT: A builtin function (`print`) passed where a `Callable[[str],
#   object]` is expected. The refusal comes from the lowering ("cannot adapt
#   builtins.function to runtime input 0"), not from the emitter.
# KIND: 未対応 (unsupported) -- refused only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a builtin passed where a callable value is expected becomes a
#   forwarder typed by that callable
#   (cases/a_builtin_function_passed_as_a_callable_value).
# CLASSIFICATION @ lang-gaps 3fa9a5fb + the fix: 1 正しい
# CPython 3.14 expects: hi
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


apply(print, "hi")
