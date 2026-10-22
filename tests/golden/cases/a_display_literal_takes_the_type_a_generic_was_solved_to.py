# WHAT: a tuple, list and dict display passed beside a value of a wider
# element type to one generic parameter (`same(groups(), ('a', None))`, the
# shape of `assertEqual(m.groups(), ...)`), to a method and to a function, and
# tuple displays returned where a homogeneous tuple is declared.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the display was built
# before the type parameter was solved and kept its own narrower type
# (`tuple[str, None]` for `tuple[str | None, ...]`), so the call was refused
# against its own formal; `-> tuple[str, ...]` refused `return ('a', 'b')`.
# That the values compared and returned are the ones written is seen in what
# is printed.
class Checker:
    def same[V](self, first: V, second: V) -> bool:
        return first == second


def same[V](first: V, second: V) -> bool:
    return first == second


def groups() -> tuple[str | None, ...]:
    return ('a', None)


def names() -> list[str | None]:
    return ['a', None]


def mapping() -> dict[str, str | None]:
    return {'k': 'v', 'n': None}


def pair() -> tuple[str, ...]:
    return ('x', 'y')


def counted() -> tuple[int | None, ...]:
    return (1, None, 3)


def make() -> list[object]:
    return [1, 'x']


def first[T](a: list[T], b: list[T]) -> T:
    return a[0]


c = Checker()
print(c.same(groups(), ('a', None)), c.same(groups(), ('a', 'b')))
print(c.same(names(), ['a', None]), c.same(mapping(), {'k': 'v', 'n': None}))
print(same(groups(), ('a', None)), same(mapping(), {'k': 'v'}))
print(pair(), counted(), groups())
print(first(make(), ['a']))
