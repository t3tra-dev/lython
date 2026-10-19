# WHAT: A tuple display holding two `Box | None` call results is stopped by
#   the ownership verifier: "conditionally owned resource ... is used before
#   its union tag proves the payload active".
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- storing a union member reads each member's class word before
#   the tag picks, and that load is not a use of the inactive payload
#   (cases/an_optional_instance_boxed_into_object takes the same path).
# CLASSIFICATION @ lang-gaps 326fa275 + the fix: 1 正しい
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
