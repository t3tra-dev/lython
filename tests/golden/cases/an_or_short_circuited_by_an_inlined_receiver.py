# WHAT: `x is None or ...` and `x is not None and ...` where x is a class
# attribute a subclass sets to None, read through an inlined method -- the
# `max_diff is None or len(diff) <= max_diff` of unittest's _truncateMessage.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: on the subclass the first
# operand is the constant True, and the second -- an int compared with None
# -- was still emitted and refused ("__le__ ... no signature that accepts
# (None)") for a branch CPython never takes. Which branch each receiver
# takes is seen in what it returns.
class Base:
    limit: int | None = 10

    def check(self, n: int) -> str:
        m = self.limit
        if m is None or n <= m:
            return "short"
        return "long"

    def strict(self, n: int) -> bool:
        m = self.limit
        r = m is not None and n > m
        return r


class NoLimit(Base):
    limit = None


class Small(Base):
    limit = 2


print(Base().check(5), NoLimit().check(50), Small().check(5))
print(Base().strict(50), NoLimit().strict(50), Small().strict(1))
