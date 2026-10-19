# WHAT: a generic method's `*items: T` and `**parts: V` are solved by what the
# call passes there -- spelled out, or as `m(*xs)` and `m(**d)` with a list, a
# `tuple[T, ...]` or a `dict[str, T]`, alone, after explicit arguments, beside
# keywords, members of different types included -- even where nothing else in
# the signature names T or V, and the body reads them at what was passed.
#
# WHY THIS IS RUN: how many members a sequence has and which keys a mapping
# holds are run-time facts.
class Bag:
    def __init__(self, name: str) -> None:
        self.name = name

    def collect[T](self, *items: T) -> list[T]:
        return list(items)

    def count[T](self, *items: T) -> int:
        return len(items)

    def label[V](self, **parts: V) -> str:
        return self.name + ":" + ",".join(k + "=" + repr(v)
                                          for k, v in parts.items())

    def lines[V](self, **parts: V) -> list[str]:
        out: list[str] = []
        for k, v in parts.items():
            out.append(k + str(v))
        return out

    def head[T](self, first: T, *rest: T) -> tuple[T, int]:
        return first, len(rest)


xs = [1, 2, 3]
words = ("a", "b")
fs = [1.5, 2.5]
d = {"x": 1.5, "y": 2.5}
b = Bag("b")
print(b.collect(*xs), b.collect(0, *xs), b.collect(*words), b.collect(1, 2))
print(b.count(*xs), b.count("p", "q"), b.collect(0, *fs), b.collect("s", *fs))
print(b.label(**d), b.label(**d, z=3.5), b.label(w=0.5), b.label(z=1, **d))
print(b.lines(**d), b.lines(k=True), b.label(**d, t=True))
print(b.head(9, *xs), b.head("q", *words))
try:
    b.label(x=0.0, **d)
except TypeError as e:
    print("TypeError:", e)
