# WHAT: A generic method returns `Box(v)` for `v: T`, annotated `-> Box[T]`.
#   The construction is not inferred as Box[T] ("annotated to return
#   Box$spec0 but this return gives Box"); `Box[T](v)` written out compiles.
#   A generic function fails the same way.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   function is annotated to return !py.contract<"Box$spec0"> but this return gives !py.contract<"Box">
# CPython 3.14 expects: 3
class Box[T]:
    def __init__(self, v: T) -> None:
        self.v = v


class Case:
    def make[T](self, v: T) -> Box[T]:
        return Box(v)


print(Case().make(3).v)
