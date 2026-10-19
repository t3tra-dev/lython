# WHAT: A tuple display holding two `Box | None` call results is stopped by
#   the ownership verifier: "conditionally owned resource ... is used before
#   its union tag proves the payload active".
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   conditionally owned resource from @_ZN8__main__5maybeE result 1 is used before its union tag proves the payload active
# CPython 3.14 expects: True True
class Box:
    def __init__(self, v: int) -> None:
        self.v = v


def maybe(n: int) -> Box | None:
    if n > 0:
        return Box(n)
    return None


t = (maybe(1), maybe(0))
print(t[0] is not None, t[1] is None)
