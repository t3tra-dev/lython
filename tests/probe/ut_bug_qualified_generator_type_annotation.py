# WHAT: `types.GeneratorType[int, None, None]` written with the module
#   prefix is not resolved (the variable is typed object); the bare name
#   imported from types works.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.contract<"builtins.object"> does not provide manifest method '__next__'
# CPython 3.14 expects: 1
import types


def gen():
    yield 1


g: types.GeneratorType[int, None, None] = gen()
print(next(g))
