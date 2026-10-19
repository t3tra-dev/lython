# WHAT: A main-module subclass of an imported class overrides a method the
#   inherited body calls. The inherited body is compiled again for the
#   subclass, in THIS module's scope, so the function it names from its own
#   module is an "unresolved name '_decorate'". Without the override it
#   works. TestCase.run calls setUp, which every test class overrides.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   unresolved name '_decorate'
# CPython 3.14 expects: <sub> <base>
import ut_mod_inherited_method


class Sub(ut_mod_inherited_method.Base):
    def name(self) -> str:
        return "sub"


print(Sub().run(), ut_mod_inherited_method.Base().run())
