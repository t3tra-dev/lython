# WHAT: A generic method `same[T](a: set[T], b: set[T])` called with two set
#   displays: "this call does not determine the type arguments". List
#   displays solve T. unittest's assertSetEqual is this method.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a set display ignores an expectation that still names a
#   type parameter, as a list display does.
# CLASSIFICATION @ lang-gaps e3a2bbfe + the fix: 1 正しい
# CPython 3.14 expects: False True
class Case:
    def same[T](self, a: set[T], b: set[T]) -> bool:
        return a == b

    def same_list[T](self, a: list[T], b: list[T]) -> bool:
        return a == b


c = Case()
print(c.same_list([1, 2], [2, 1]), c.same({1, 2}, {2, 1}))
