# WHAT: a subclass here overrides `name`, which the imported base's `run` and
# `__repr__` call, so both are compiled again with `self` at the subclass --
# and each copy still reads the module that wrote it: the helper function and
# the global private to that module. A list's repr reaches the copy through
# its symbol, a call through the dispatch over the two classes.
#
# WHY THIS IS RUN: which `name` a copy calls is decided by the receiver's
# class at run time. The copies were compiled in this module, where
# `_decorate` was an "unresolved name".
from an_inherited_body_copied_for_a_subclass_reads_its_own_module_lib import Base


class Sub(Base):
    def name(self) -> str:
        return "sub"


items: list[Base] = [Sub(), Base()]
for item in items:
    print(item.run())
print(items)
print(Sub().run(), [Sub()])
