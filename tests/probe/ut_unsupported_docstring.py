# WHAT: `f.__doc__`: docstrings are not kept at run time. unittest's
#   shortDescription() prints a test's first docstring line under -v.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a docstring is answered where it is read
#   (cases/a_docstring_is_answered_where_it_is_read). The read is
#   `str | None`, as typeshed's `__doc__` is, so it is narrowed before
#   `.splitlines()`: the call on the read itself was ill-typed.
# CLASSIFICATION @ lang-gaps 6a9290f8 + the fix: 1 正しい
# CPython 3.14 expects: First line.
class C:
    def test_a(self) -> None:
        """First line.

        More."""


doc = C.test_a.__doc__
print(doc.splitlines()[0] if doc else None)
