# WHAT: a method of this module wrapped by decorators another module
# defines -- a class whose `__call__` is generic, and a generic function --
# runs what they wrapped it in.
#
# WHY THIS IS RUN: the wrapper is a def nested in the imported generic
# method, made when the decorator runs; it was made with its type parameter
# unsolved, and calling it raised "callable target is not available".
from an_imported_generic_decorator_wraps_a_method_lib import logged, skip


class Case:
    def __init__(self, name: str) -> None:
        self.name = name

    @skip("not today")
    def test_a(self) -> None:
        print("ran a", self.name)

    @logged
    def test_b(self) -> None:
        print("ran b", self.name)


c = Case("x")
c.test_a()
c.test_b()
