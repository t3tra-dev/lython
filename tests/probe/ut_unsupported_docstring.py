# WHAT: `f.__doc__`: docstrings are not kept at run time. unittest's
#   shortDescription() prints a test's first docstring line under -v.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   '__doc__' is not available: docstrings are not retained at run time
# CPython 3.14 expects: First line.
class C:
    def test_a(self) -> None:
        """First line.

        More."""


print(C.test_a.__doc__.splitlines()[0])
