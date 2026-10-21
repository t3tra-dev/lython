# WHAT: A module-level class whose base is a class defined further down.
#   CPython raises NameError at the class statement; here the program runs,
#   because a class is a module-level contract bound before any statement
#   executes.
# KIND: バグ (bug) -- permissive: runs where CPython raises
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- refused at the base, before anything runs: the order a
#   class statement reads its bases in is the source's.
# CLASSIFICATION @ lang-gaps e77e1398 + the fix: 4 両方が拒否
# CPython 3.14 expects: (例外を送出)
class A(Later):
    pass


class Later:
    pass


print("defined", A.__name__)
