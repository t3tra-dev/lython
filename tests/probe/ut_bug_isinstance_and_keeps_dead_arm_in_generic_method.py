# WHAT: In a generic method specialized at int, `isinstance(a, list) and
#   isinstance(b, list)` is False, yet the arm under it is still type-
#   checked ("is declared list[int] and this call gives it int"). One
#   isinstance alone prunes the arm; so do two nested ifs. unittest's
#   assertEqual dispatches on both arguments' types this way.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   argument 'a' of 'listy' is declared !py.contract<"builtins.list", [!py.contract<"builtins.int">]> and this call gives it !py.contract<"builtins.int">
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
