# WHAT: a class from another module has classmethods, and they are reached
# through an instance from this module: by a computed name inside the other
# module's runner -- whose arms are written while this module is being
# compiled -- and by plain attribute calls on instances of the base, of its
# subclass there and of a subclass here. Each binds `cls` to the instance's
# class at run time.
#
# WHY THIS IS RUN: the class a classmethod is bound to through an instance is
# the instance's class at run time. The runner's arm was re-spelled as the
# bare `TestCase.setUpClass()`, a name only the other module binds ("unresolved
# name 'TestCase'"), and `a.kind()` below was re-spelled the same way.
import a_computed_name_call_reaches_a_classmethod_across_modules_lib as lib


class MyTests(lib.TestCase):
    def test_a(self) -> None:
        print("test_a ran")


lib.run_case(MyTests("test_a"))
lib.run_case(MyTests("setUpClass"))
lib.run_case(lib.TestCase("setUpClass"))
a: lib.TestCase = lib.Special()
print(a.kind(), lib.TestCase().kind(), MyTests().kind())
