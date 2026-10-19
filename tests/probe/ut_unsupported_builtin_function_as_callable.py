# WHAT: A builtin function (`print`) passed where a `Callable[[str],
#   object]` is expected. The refusal comes from the lowering ("cannot adapt
#   builtins.function to runtime input 0"), not from the emitter.
# KIND: 未対応 (unsupported) -- refused only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   cannot adapt builtins.function to runtime input 0 of builtins.function._ZN8__main__5applyE [values:, expected 'memref<8xi64>']
# CPython 3.14 expects: hi
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


apply(print, "hi")
