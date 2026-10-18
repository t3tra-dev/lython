# WHAT: a class whose qualified name is longer than 64 bytes is named whole by
# `type(v).__name__`, `v.__class__.__name__` and the default repr.
#
# WHY THIS IS RUN: the value's static type is its base, so each name is read
# off the class's type object at run time; the readers copied at most 64
# bytes of it, and printed what fit as the class.
class Base:
    pass


class AVeryLongClassNameThatGoesOnAndOnForMoreThanSixtyFourCharactersX(Base):
    pass


def pick(flag: bool) -> Base:
    if flag:
        return AVeryLongClassNameThatGoesOnAndOnForMoreThanSixtyFourCharactersX()
    return Base()


b = pick(True)
print(type(b).__name__)
r = repr(b)
print(r[: r.index(" object at")])
print([b][0].__class__.__name__)
