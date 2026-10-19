# WHAT: A module-level class whose base is a class defined further down.
#   CPython raises NameError at the class statement; here the program runs,
#   because a class is a module-level contract bound before any statement
#   executes.
# KIND: バグ (bug) -- permissive: runs where CPython raises
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython raised, lyc completed cleanly
# CPython 3.14 expects: (例外を送出)
class A(Later):
    pass


class Later:
    pass


print("defined", A.__name__)
