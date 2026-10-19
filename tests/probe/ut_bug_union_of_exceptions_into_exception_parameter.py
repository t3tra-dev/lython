# WHAT: A `ValueError | KeyError` value passed to an `Exception` parameter:
#   the emitter accepts it, the lowering stops with "cannot adapt to runtime
#   input 0".
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   cannot adapt  to runtime input 0 of builtins.function._ZN8__main__8describeE [values: 'i64' 'memref<3xi64>' 'memref<2xi64>' 'memref<?xi8>' 'memref<3xi64>' 'memref<2xi64>' 'memref<?xi8>', expected 'memref<3xi64>']
# CPython 3.14 expects: ValueError: v KeyError: 'k'
def describe(e: Exception) -> str:
    return type(e).__name__ + ": " + str(e)


def pick(flag: bool) -> ValueError | KeyError:
    if flag:
        return ValueError("v")
    return KeyError("k")


print(describe(pick(True)), describe(pick(False)))
