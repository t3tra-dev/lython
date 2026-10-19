# WHAT: A TypeError raised by __hash__ during a dict store. CPython 3.14
#   wraps it: "cannot use 'Key' as a dict key (nope)"; here the raw message
#   arrives.
# KIND: バグ (bug) -- silent: the message differs
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong the dict_unhashable_type wrapping of CPython 3.14 is missing
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython="TypeError: cannot use 'Key' as a dict key (nope)\n" lyc='TypeError: nope\n'
# CPython 3.14 expects: TypeError: cannot use 'Key' as a dict key (nope)
class Key:
    def __hash__(self) -> int:
        raise TypeError("nope")


d: dict[Key, int] = {}
try:
    d[Key()] = 1
except TypeError as e:
    print("TypeError:", e)
