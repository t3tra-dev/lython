# WHAT: A fixed tuple passed to a `tuple[int, ...]` parameter of a GENERATOR
#   METHOD is refused ("call arguments do not match the Callable contract");
#   a free function or a plain method accepts it.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- a generator method's arguments are emitted at its
#   parameters' types, as any call's are.
# CLASSIFICATION @ lang-gaps 1af75821 + the fix: 1 正しい
# CPython 3.14 expects: [1, 2, 3]
from collections.abc import Iterator


class G:
    def walk(self, xs: tuple[int, ...]) -> Iterator[int]:
        for x in xs:
            yield x


print(list(G().walk((1, 2, 3))))
