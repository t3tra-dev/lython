# WHAT: repr() of a generator names the generator's FUNCTION, as CPython's
#   <generator object gen at 0x..> does; here it names the class that holds
#   the frame, <__main__.GeneratorType object at 0x..>. The address is cut
#   off so the two runs can be compared.
# KIND: バグ (bug) -- silent: prints the wrong text
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- the runtime's default repr asks the lowering for the
#   generator body's qualname by its target id; the class is `generator`.
# CLASSIFICATION @ lang-gaps 0c48833a + the fix: 1 正しい
# CPython 3.14 expects: <generator object gen
def gen():
    yield 1


print(repr(gen()).split(" at ")[0])
