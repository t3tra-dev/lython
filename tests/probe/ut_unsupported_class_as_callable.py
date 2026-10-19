# WHAT: A class passed where a `Callable[[int], Box]` is expected (its
#   constructor as a value).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.callable<[!py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"Box">]>, !py.contract<"builtins.int">], arg_names = ["f", "v"], arg_defaults = [false, false], returns = [!py.contract<"Box"
# CPython 3.14 expects: 3
from collections.abc import Callable


class Box:
    def __init__(self, v: int) -> None:
        self.v = v


def build(f: Callable[[int], Box], v: int) -> Box:
    return f(v)


print(build(Box, 3).v)
