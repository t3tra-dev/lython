# WHAT: an object with a `type[X] | None` field -- X with a subclass, and X
# with none -- held by another object and by a list, the field written
# through both and read back by identity.
#
# WHY THIS IS RUN: which class the field holds is a run-time fact. The field
# was kept inline in its object, which then could not be stored in another
# object or a list ("expands to 3 physical values").
class Case:
    pass


class Sub(Case):
    pass


class Lone:
    pass


class Result:
    def __init__(self) -> None:
        self.n = 0
        self.previous: type[Case] | None = None
        self.lone: type[Lone] | None = None


class Outcome:
    def __init__(self, result: Result) -> None:
        self.result = result


o = Outcome(Result())
print(o.result.previous is None, o.result.lone is None)
o.result.previous = Sub
o.result.lone = Lone
print(o.result.previous is Sub, o.result.previous is Case, o.result.lone is Lone)
results = [Result(), Result()]
results[1].previous = Case
print([r.previous is Case for r in results], [r.previous is None for r in results])
