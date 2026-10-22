# WHAT: a generic function asking isinstance of a container its caller filled
# with nothing -- `[]`, `()`, `{}`, `set()` are containers of `object` -- and
# a function narrowing an `object` to each container kind, frozenset too.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the first was refused
# ("isinstance on an object-typed value requires dynamic object inspection")
# although the container's own class is known, and the frozenset arm read a
# frozenset with no element type ("does not provide '__len__'"). What has to
# hold is the arm each value takes and what is read in it, at run time for
# the `object` ones.
def kind[T](x: T) -> str:
    if isinstance(x, dict):
        return "dict %d" % len(x)
    if isinstance(x, list):
        return "list %d" % len(x)
    if isinstance(x, tuple):
        return "tuple %d" % len(x)
    if isinstance(x, frozenset):
        return "frozenset %d" % len(x)
    if isinstance(x, set):
        return "set %d" % len(x)
    return "other"


def size(o: object) -> str:
    if isinstance(o, frozenset):
        return "frozenset of %d" % len(o)
    if isinstance(o, set):
        return "set of %d" % len(o)
    if isinstance(o, list):
        return "list of %d" % len(o)
    return repr(o)


print(kind([]), kind(()), kind({}), kind(set()), kind([1]), kind(frozenset({2})))
values: list[object] = [frozenset({1, 2}), {3}, [4, 5, 6], "s"]
print([size(v) for v in values])
