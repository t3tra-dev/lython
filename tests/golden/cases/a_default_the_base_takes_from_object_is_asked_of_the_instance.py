# WHAT: on a value typed by a base class that declares no `__repr__`,
# `__str__` or `__hash__`, a subclass's override answers for an instance of
# that subclass -- through `repr()`, `str()`, `hash()`, f-strings and
# %-formatting alike -- and object's default answers for the base itself.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which class answers is
# the instance's run-time class. The base had no body to fall back on, so
# these were refused ("'__repr__' is overridden by a subclass of 'Base'"),
# and `str(o)` over a subclass declaring only `__str__` printed the default
# repr where CPython calls the subclass's `__str__`.


class Base:
    pass


class Reprs(Base):
    def __repr__(self) -> str:
        return "R:" + type(self).__name__


class Strs(Base):
    def __str__(self) -> str:
        return "S:" + type(self).__name__


class Both(Base):
    def __repr__(self) -> str:
        return "R:" + type(self).__name__

    def __str__(self) -> str:
        return "S:" + type(self).__name__


class Hashes(Base):
    def __hash__(self) -> int:
        return 7


class Leaf(Both):
    pass


def without_addresses(text: str) -> str:
    pieces = text.split(" at 0x")
    out = pieces[0]
    for piece in pieces[1:]:
        out += piece[piece.index(">"):]
    return out


def shown(o: Base) -> str:
    return without_addresses(" ".join([
        repr(o), str(o), f"{o}", f"{o!r}", "%s" % (o,), "%r" % (o,),
        str(hash(o) == 7)]))


for o in [Base(), Reprs(), Strs(), Both(), Hashes(), Leaf()]:
    print(shown(o))
