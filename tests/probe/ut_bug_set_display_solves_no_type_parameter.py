# WHAT: A generic method `same[T](a: set[T], b: set[T])` called with two set
#   displays: "this call does not determine the type arguments". List
#   displays solve T. unittest's assertSetEqual is this method.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   this call does not determine the type arguments of the generic method 'same'
# CPython 3.14 expects: False True
class Case:
    def same[T](self, a: set[T], b: set[T]) -> bool:
        return a == b

    def same_list[T](self, a: list[T], b: list[T]) -> bool:
        return a == b


c = Case()
print(c.same_list([1, 2], [2, 1]), c.same({1, 2}, {2, 1}))
