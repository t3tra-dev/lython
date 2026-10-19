# WHAT: A fixed tuple passed to a `tuple[int, ...]` parameter of a GENERATOR
#   METHOD is refused ("call arguments do not match the Callable contract");
#   a free function or a plain method accepts it.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.callable<[!py.contract<"G">, !py.contract<"builtins.tuple", [!py.contract<"builtins.int">]>], arg_names = ["self", "xs"], arg_defaults = [false, false], returns = [!py.contract<"types.GeneratorType", [!py
# CPython 3.14 expects: [1, 2, 3]
from collections.abc import Iterator


class G:
    def walk(self, xs: tuple[int, ...]) -> Iterator[int]:
        for x in xs:
            yield x


print(list(G().walk((1, 2, 3))))
