# WHAT: A function whose only yield sits under `if False:` is still a
#   generator in CPython (an empty one). Here the yield is folded away
#   before the generator is built, and the lowering refuses a generator with
#   no yield ("requires at least one yield"); held in a field, the same
#   function is typed object.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   source generator next lowering currently requires at least one yield
# CPython 3.14 expects: []
def empty():
    if False:
        yield 1


print(list(empty()))
