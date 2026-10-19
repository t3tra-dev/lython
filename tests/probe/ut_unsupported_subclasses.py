# WHAT: `cls.__subclasses__()`. The program's classes are all known when it
#   is compiled, so the answer is static; it would let `unittest.main()`
#   find the TestCase subclasses without module reflection.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- answered from the program's classes, in creation order
#   (cases/a_class_lists_its_subclasses).
# CLASSIFICATION @ lang-gaps 18111c5c + the fix: 1 正しい
# CPython 3.14 expects: ['A', 'B'] ['C']
class Base:
    pass


class A(Base):
    pass


class B(Base):
    pass


class C(A):
    pass


print([c.__name__ for c in Base.__subclasses__()], [c.__name__ for c in A.__subclasses__()])
