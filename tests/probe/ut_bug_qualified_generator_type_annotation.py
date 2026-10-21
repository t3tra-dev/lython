# WHAT: `types.GeneratorType[int, None, None]` written with the module
#   prefix is not resolved (the variable is typed object); the bare name
#   imported from types works.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- a subscripted annotation whose base is a dotted name
#   resolves the base as the unsubscripted annotation does.
# CLASSIFICATION @ lang-gaps 08e9f008 + the fix: 1 正しい
# CPython 3.14 expects: 1
import types


def gen():
    yield 1


g: types.GeneratorType[int, None, None] = gen()
print(next(g))
