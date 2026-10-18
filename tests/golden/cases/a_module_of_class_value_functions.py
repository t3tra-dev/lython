# WHAT: a module the next case imports -- functions that take a class value
# of a base class this module declares, and a class test on an instance.
# Run alone it prints nothing.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the imported half of
# a_class_value_names_a_subclass_another_module_declares.py, which needs it on
# disk; running it alone keeps it a program of its own.
class Base:
    def __init__(self, n: int) -> None:
        self.n = n

    def who(self) -> str:
        return "base"


def name_of(cls: type[Base]) -> str:
    return cls.__name__


def build(cls: type[Base]) -> Base:
    return cls(1)


def same(cls: type[Base]) -> bool:
    return cls is Base


def kind(b: Base) -> str:
    return type(b).__name__


def check(b: Base) -> bool:
    return type(b) is Base
