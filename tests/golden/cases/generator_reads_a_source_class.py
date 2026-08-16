# What this pins: a top-level generator whose body reads a source class.
#
#     class C:
#         def __init__(self) -> None:
#             self.n = 5
#     def gen(c: C):
#         yield c.n
#     print(list(gen(C())))
#     # runtime bundle for 'builtins.object' has 5 values, but ABI expects 1
#
# Every top-level signature is computed before any class contract exists, and
# it has to be: a signature may NAME a class, and the class's own bodies are
# typed against the function symbols. For an ordinary function that order is
# harmless -- only its annotations matter. A generator's signature also depends
# on its BODY, because its yield type is inferred from what it yields, and
# `c.n` against a class with no published fields infers `builtins.object`.
#
# The same generator NESTED inside a function always worked, because its
# signature is computed during body emission, after the classes. That pair is
# what says it is the ordering and not the reading.
#
# Why this needs to run rather than assert on a diagnostic: the yield type is
# not written anywhere in the program -- it is what the CONSUMER compiles
# against. `Iterator[int]` and `Iterator[object]` both build a generator; only
# the value the loop gets says which one the frame carried, and a method call
# on the yielded value is the sharpest check.
#
# ⛔ A generator textually BEFORE the class it reads (reachable only with a
# string annotation) still gets the early answer. Emitting all classes first
# would fix that and trade it for the reverse -- a class body may reference a
# module-level function -- so source order is respected instead.
#
# Every expected line is python3.14's.


class Counter:
    def __init__(self, start: int) -> None:
        self.n = start
        self.log: list[str] = []

    def tick(self) -> int:
        self.n += 1
        self.log.append("t")
        return self.n

    @property
    def doubled(self) -> int:
        return self.n * 2


# --- a field read, a method call and a property, all in the yield ---------
def field_reads(c: Counter):
    yield c.n
    yield c.doubled


def method_calls(c: Counter, k: int):
    i = 0
    while i < k:
        yield c.tick()
        i += 1


c = Counter(5)
print(list(field_reads(c)))
print(list(method_calls(c, 3)))
print(c.n, c.log)


# --- the yielded value has to still be an int downstream ------------------
totals = [v * 2 for v in field_reads(Counter(1))]
print(totals, sum(totals))
for v in method_calls(Counter(0), 2):
    print(v + 100)


# --- a generator yielding the INSTANCE, which already worked --------------
def instances(k: int):
    i = 0
    while i < k:
        yield Counter(i)
        i += 1


print([c2.doubled for c2 in instances(3)])


# --- THE CONTROL: nested in a function, which is the shape that worked ----
def outer() -> list[int]:
    def inner(c3: Counter):
        yield c3.n

    return list(inner(Counter(9)))


print(outer())
