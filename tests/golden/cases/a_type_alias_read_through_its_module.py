# WHAT: a module's type aliases used from the importer as `m.Pair`, as
# `from m import Pair`, and as a local alias of one (`P = m.Pair`), an alias
# naming another module's class (`types.TracebackType`), and a class alias
# (`main = Program`) called through the module.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: an alias spelled through
# its module was read as a class of that name nobody declares, so the tuple was
# not unpacked ("does not provide manifest method '__getitem__'"); the program
# has to run to show the values went through the real types.
import sys

import a_type_alias_read_through_its_module_lib as m
from a_type_alias_read_through_its_module_lib import Pair

P = m.Pair


def through_module(p: m.Pair) -> str:
    a, b = p
    return b * a


def through_import(p: Pair) -> str:
    a, b = p
    return b + str(a)


def through_local(p: P) -> int:
    return p[0] + len(p[1])


def kind(info: m.Info) -> str:
    return info[0].__name__ + ": " + str(info[1])


print(through_module((2, "ab")), through_import((3, "x")), through_local((1, "yz")))
try:
    raise KeyError("k")
except KeyError:
    exc_type, exc, tb = sys.exc_info()
    if exc_type is not None and exc is not None and tb is not None:
        print(kind((exc_type, exc, tb)))
m.main()
m.main(3)
