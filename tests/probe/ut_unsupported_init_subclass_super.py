# WHAT: `__init_subclass__` calling `super().__init_subclass__()`: "zero-
#   argument super() requires an enclosing class method body". A registry
#   filled by __init_subclass__ is another way to find a module's test
#   classes.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   zero-argument super() requires an enclosing class method body
# CPython 3.14 expects: ['A', 'B']
registry: list[str] = []


class Base:
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        registry.append(cls.__name__)


class A(Base):
    pass


class B(Base):
    pass


print(registry)
