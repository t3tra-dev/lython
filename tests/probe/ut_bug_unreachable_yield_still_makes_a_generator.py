# WHAT: A function whose only yield sits under `if False:` is still a
#   generator in CPython (an empty one). Here the yield is folded away
#   before the generator is built, and the lowering refuses a generator with
#   no yield ("requires at least one yield"); held in a field, the same
#   function is typed object.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- the generator state machine takes a body with no yield
#   left, with the declared element's value lane, as it takes a coroutine
#   that never awaits.
# CLASSIFICATION @ lang-gaps a98ba3ee + the fix: 1 正しい
# CPython 3.14 expects: []
def empty():
    if False:
        yield 1


print(list(empty()))
