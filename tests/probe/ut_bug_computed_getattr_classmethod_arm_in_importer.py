# WHAT: The computed-name call in the imported runner has an arm per
#   attribute, written after every module's classes are known -- in THIS
#   module's scope, where the arm for the classmethod names `TestCase`,
#   which only the imported module binds ("unresolved name 'TestCase'").
#   `from ut_mod_computed_getattr import TestCase` here makes it compile.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a program class's classmethod reached through an instance is
#   no longer re-spelled as a call on the class's bare name.
# CLASSIFICATION @ lang-gaps 9bf68b55 + the fix: 1 正しい
# CPython 3.14 expects: test_a ran
import ut_mod_computed_getattr


class MyTests(ut_mod_computed_getattr.TestCase):
    def test_a(self) -> None:
        print("test_a ran")


ut_mod_computed_getattr.run_case(MyTests("test_a"))
