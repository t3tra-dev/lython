# What: a class that defines __eq__ and not __hash__ is unhashable, as in
#   CPython (which sets __hash__ to None for it -- every unfrozen dataclass
#   included), and the refusal says so. The alternative was to answer object's
#   identity hash now that object's defaults are inherited, which would place
#   two instances the class calls EQUAL in different hash buckets and make
#   every container built on them miss without a word.
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


print(hash(Point(1, 2)))
