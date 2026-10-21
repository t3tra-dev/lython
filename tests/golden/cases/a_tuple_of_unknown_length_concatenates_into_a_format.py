# WHAT: `(i,) + reprs` where `reprs` is a `tuple[str, ...]` handed to a `%`
# format with `%d` and `%s`, tuple `+` of two unknown lengths, and an f-string
# with no replacement field used as the `%` template.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the concatenation is how
# unittest prints the first differing element, and its element type was lost
# (`tuple` of nothing), so the format had no conversion for it; the f-string
# template was refused as not a compile-time format string. The formatted text
# is what shows each element took its own conversion.
def reprs(*args: object) -> tuple[str, ...]:
    return tuple(repr(a) for a in args)


i = 3
print('\nFirst differing element %d:\n%s\n%s' % ((i,) + reprs(1, "x")))
both = reprs("a") + reprs(2, 3)
print(len(both), both)
print(f'%s.%s.%s' % ("mod", "Class", "test_x"))
