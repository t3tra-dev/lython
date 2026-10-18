# WHAT: a class value of an imported base, holding a subclass this module
# declares, answers for the subclass inside the imported module's functions:
# its name, the instance it builds, its identity with the base, and the class
# test on an instance of it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the class is the one the
# value holds at run time. The imported functions are emitted before this
# module declares Sub, and took Base for a class nothing derives from:
# `cls.__name__` answered "Base", `cls(1)` built a Base, and both identity
# tests answered True -- silently.
from a_module_of_class_value_functions import Base, build, check, kind, name_of, same


class Sub(Base):
    def who(self) -> str:
        return "sub"


print(name_of(Sub), name_of(Base))
print(type(build(Sub)).__name__, build(Sub).who(), build(Base).who())
print(same(Sub), same(Base), kind(Sub(1)), check(Sub(1)), check(Base(1)))
