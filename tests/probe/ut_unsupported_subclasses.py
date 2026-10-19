# WHAT: `cls.__subclasses__()`. The program's classes are all known when it
#   is compiled, so the answer is static; it would let `unittest.main()`
#   find the TestCase subclasses without module reflection.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.type<!py.contract<"Base">> does not provide manifest method '__subclasses__'
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
