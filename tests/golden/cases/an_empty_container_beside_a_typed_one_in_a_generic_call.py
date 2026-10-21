# WHAT: an empty list, dict and set passed to a generic function and to a
# generic method beside a typed one of the same kind, in either order, and the
# empty one then filled inside the call.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the empty literal took part
# in solving T as `list[object]` -- the function's call died in the lowering
# ("arguments do not match Callable contract") and the method's was refused --
# and whether the empty argument became the same type as T is only seen when
# the body puts an element in it and the program reads it back.
def same[T](a: T, b: T) -> bool:
    return a == b


def fill[T](into: list[T], source: list[T]) -> list[T]:
    for item in source:
        into.append(item)
    return into


class Checker:
    def same[T](self, a: T, b: T) -> bool:
        return a == b


log: list[str] = []
counts: dict[str, int] = {}
seen: set[int] = set()
print(same(log, []), same([], log), same(counts, {}), same(seen, set()))
print(Checker().same(log, []), Checker().same([], log), Checker().same(counts, {}))
filled = fill([], ["a", "b"])
print(filled, len(filled[0] + filled[1]))
log.append("x")
print(same(log, []), Checker().same(["x"], log))
