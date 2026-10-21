# WHAT: A `ValueError | KeyError` value passed to an `Exception` parameter:
#   the emitter accepts it, the lowering stops with "cannot adapt to runtime
#   input 0".
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- a union of classes below the declared class is the
#   active member upcast, one arm per member, wherever it is converted.
# CLASSIFICATION @ lang-gaps 70b81ceb + the fix: 1 正しい
# CPython 3.14 expects: ValueError: v KeyError: 'k'
def describe(e: Exception) -> str:
    return type(e).__name__ + ": " + str(e)


def pick(flag: bool) -> ValueError | KeyError:
    if flag:
        return ValueError("v")
    return KeyError("k")


print(describe(pick(True)), describe(pick(False)))
