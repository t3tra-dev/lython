# A manifest class reaching its base's implementations. Execution is needed
# because resolution and adaptation are separate halves and only the values show
# both ran: picking int's method proves the base walk, and printing 1 rather
# than a garbage word proves the truth bit was widened into int's lanes on the
# way in. A compile-only check would pass with either half broken.
#
# __invert__ is deliberately absent even though it works: CPython 3.14 warns
# that ~bool is deprecated, and pinning it would make this file a 3.16 failure
# about something it is not testing.

flag: bool = True
off: bool = False

print(abs(flag), abs(off))
print(abs(True), abs(False))
print(flag.__int__(), off.__int__())
print(flag.__index__(), off.__index__())
print(flag.__round__(0), off.__round__(0))
print(round(flag, 0))

# The int spellings, so a repair that reroutes the base's own callers is caught
# in the same file.
n: int = -7
print(abs(n), n.__int__(), n.__index__(), n.__round__(0))
