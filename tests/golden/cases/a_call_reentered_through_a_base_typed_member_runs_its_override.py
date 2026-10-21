# WHAT: a suite whose `__call__` runs `self.run`, holding suites and cases as
# a union; the outer suite runs, and its loop calls each member -- a nested
# suite reaches `Base.__call__` again while it is already being expanded. The
# nested suite is a `Suite`, whose `run` overrides the base's.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the recursive call took
# the body compiled for a receiver of exactly `Base`, so the nested suite ran
# `Base.run` -- ['suite run', 'base run', 'case'] -- with nothing refused.
class Case:
    def __call__(self, r: list[str]) -> list[str]:
        r.append("case")
        return r


class Base:
    def __init__(self) -> None:
        self.items: list[Case | Base] = []

    def run(self, r: list[str]) -> list[str]:
        r.append("base run")
        for t in self.items:
            t(r)
        return r

    def __call__(self, r: list[str]) -> list[str]:
        return self.run(r)


class Suite(Base):
    def run(self, r: list[str], debug: bool = False) -> list[str]:
        r.append("suite run")
        for t in self.items:
            t(r)
        return r


inner = Suite()
inner.items.append(Case())
plain = Base()
plain.items.append(Case())
outer = Suite()
outer.items.append(inner)
outer.items.append(plain)
print(outer([]))
