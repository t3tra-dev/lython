# WHAT: A main-module subclass calls an inherited generic method of an
#   imported class, which builds an imported generic class whose __init__
#   annotates `self._exception: E | None`. The annotation is read with E
#   unbound ("attribute value 'builtins.E | None' is not assignable to field
#   'ValueError | None'"); called on the imported class itself, or with the
#   field declared in the class body, it compiles. Same family as
#   ut_bug_inherited_method_loses_its_module_globals.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   attribute value '!py.union<!py.contract<"builtins.E">, !py.literal<None>>' is not assignable to field '!py.union<!py.contract<"builtins.ValueError">, !py.literal<None>>'
# CPython 3.14 expects: True False
import ut_mod_generic_context


class Mine(ut_mod_generic_context.Case):
    def test(self) -> None:
        c = self.raises(ValueError)
        print(c.take(ValueError("v")), c.take(KeyError("k")))


Mine().test()
