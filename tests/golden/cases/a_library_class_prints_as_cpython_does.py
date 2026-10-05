# WHAT: classes of the runtime's Python library that CPython prints in a form
# of their own -- Counter (most common first, and a subclass under its own
# name), OrderedDict and a subclass of it, and time.struct_time -- printed
# by print, repr, str and inside a list.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was the text a
# running program printed. Counter and struct_time had no __repr__ in the
# port, so they printed the default `<collections.Counter object at 0x...>`,
# and OrderedDict's named itself whatever the subclass. os.stat_result is
# checked as far as its field names go; the values are the machine's.
import os
import time
from collections import Counter, OrderedDict

c = Counter("abracadabra")
print(c)
print(repr(c), str(c))
print([c])
print(Counter())
print(c.most_common(2))


class Tally(Counter):
    pass


print(Tally("aab"))
od: OrderedDict[str, int] = OrderedDict()
od["b"] = 1
od["a"] = 2
print(od, OrderedDict[str, int]())


class Ordered(OrderedDict[str, int]):
    pass


o = Ordered()
o["k"] = 3
print(o)
print(time.gmtime(0))
print([time.gmtime(1700000000)])
print(repr(os.stat("/"))[:23])
