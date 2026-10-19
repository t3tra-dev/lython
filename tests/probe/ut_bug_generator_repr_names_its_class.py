# WHAT: repr() of a generator names the generator's FUNCTION, as CPython's
#   <generator object gen at 0x..> does; here it names the class that holds
#   the frame, <__main__.GeneratorType object at 0x..>. The address is cut
#   off so the two runs can be compared.
# KIND: バグ (bug) -- silent: prints the wrong text
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong the repr names GeneratorType, not the function
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython='<generator object gen\n' lyc='<__main__.GeneratorType object\n'
# CPython 3.14 expects: <generator object gen
def gen():
    yield 1


print(repr(gen()).split(" at ")[0])
