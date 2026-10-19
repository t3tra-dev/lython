# WHAT: `main([Mine])` -- a one-element list of a main-module subclass
#   passed to an IMPORTED function's `list[type[TestCase]]` parameter. The
#   emitter accepts it, the lowering stops with "arguments do not match
#   Callable contract". Two classes (`[Mine, Other]`) join to the base and
#   compile, as does the same call within one module.
#   `unittest.main([MyTests])` is this call.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a subclass's class value fits its base's class type in a
#   container display's expectation.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
# CPython 3.14 expects: ['Mine']
import ut_mod_list_of_classes


class Mine(ut_mod_list_of_classes.TestCase):
    pass


ut_mod_list_of_classes.main([Mine])
