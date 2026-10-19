# WHAT: In a generic method specialized at int, `isinstance(a, list) and
#   isinstance(b, list)` is False, yet the arm under it is still type-
#   checked ("is declared list[int] and this call gives it int"). One
#   isinstance alone prunes the arm; so do two nested ifs. unittest's
#   assertEqual dispatches on both arguments' types this way.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- an `and`/`or` decided by its operands decides the branch;
#   a later decided operand after run-time ones prunes the dead arm too.
# CLASSIFICATION @ lang-gaps e3a2bbfe + the fix: 1 正しい
# CPython 3.14 expects: other 1 list True
class T:
    def listy(self, a: list[int], b: list[int]) -> str:
        return "list " + str(a == b)

    def check[V](self, first: V, second: V) -> str:
        if isinstance(first, list) and isinstance(second, list):
            return self.listy(first, second)
        return "other " + repr(first)


t = T()
print(t.check(1, 2), t.check([1, 2], [1, 2]))
