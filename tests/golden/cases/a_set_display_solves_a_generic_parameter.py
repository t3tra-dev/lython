# WHAT: a set display written as the argument of a generic function and of a
# generic method solves the type parameter from its elements, as a list
# display and a set held in a name do; the specialization then compares,
# measures and differences the sets.
#
# WHY THIS IS RUN: the differences are computed at run time. The display was
# built at the parameter's own `set[T]`, which left T with nothing to solve
# from ("this call does not determine the type arguments").
def same[T](a: set[T], b: set[T]) -> bool:
    return a == b


def size[T](a: set[T]) -> int:
    return len(a)


class Case:
    def missing[T](self, a: set[T], b: set[T]) -> list[T]:
        return sorted(a - b)

    def same_list[T](self, a: list[T], b: list[T]) -> bool:
        return a == b


x = {1, 2}
print(size(x), size({1, 2, 2}), same({1, 2}, {2, 1}), same({"a"}, {"b"}))
c = Case()
print(c.same_list([1, 2], [2, 1]), c.missing({1, 2, 3}, {2}), c.missing({"x", "y"}, {"y"}))
