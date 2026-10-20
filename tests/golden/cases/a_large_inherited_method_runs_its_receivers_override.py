# WHAT: a large method of an imported class, called through a receiver whose
# class the call site does not know, runs the hook of the object's own class --
# and so does one called on an object of a class that inherits it; an omitted
# argument takes its default, and a keyword fills a union parameter.
#
# WHY THIS IS RUN: which `hook` the body reaches is the receiver's class at run
# time.
from a_large_inherited_method_runs_its_receivers_override_lib import Base, Mid


class Leaf(Mid):
    def hook(self, i: int) -> int:
        return i * 100


class Other(Base):
    pass


def through_base(x: Base, label: str) -> int:
    return x.report(label)


def through_mid(x: Mid, label: str) -> int:
    return x.report(label, limit=3)


print(Base("b").report("exact"))
print(through_base(Leaf("l"), "base-typed"))
print(through_base(Mid("m"), "base-typed-mid"))
print(through_mid(Leaf("l2"), "mid-typed"))
print(Mid("m2").report("joined", "-"))
print(Other("o").report("inherited", limit=None))
print(through_base(Other("o2"), "other"))
