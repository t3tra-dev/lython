# `max(items)` / `min(items)` over instances that order themselves. sorted()
# beside them has always worked; these were refused for a seed they do not
# need -- "max() needs an element type the fold can seed (int, str, float,
# bool, or a tuple of those), or an indexable argument to take the first
# element from", offered about a list, which is indexable.
#
# Golden because the fold is a rewrite: it walks the list by index and reads
# the winner once at the end, and CPython's tie rule (the FIRST maximal
# element) is a value question, not a compile one -- hence the `is` checks on
# the duplicate pair.
class Version:
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor

    def __lt__(self, other: "Version") -> bool:
        if self.major != other.major:
            return self.major < other.major
        return self.minor < other.minor

    def __gt__(self, other: "Version") -> bool:
        if self.major != other.major:
            return self.major > other.major
        return self.minor > other.minor

    def __repr__(self) -> str:
        return str(self.major) + "." + str(self.minor)


versions = [Version(1, 2), Version(1, 0), Version(2, 0), Version(1, 10)]
print(sorted(versions))
print(max(versions), min(versions))

tied = [Version(3, 0), Version(3, 0), Version(1, 0)]
print(max(tied) is tied[0], min(tied) is tied[2])

print(max(versions, key=lambda v: v.minor), min(versions, key=lambda v: v.minor))
print(max([], default=None), min([], default=Version(9, 9)))
try:
    max([])
except ValueError as e:
    print("ValueError:", e)
print(max([Version(1, 1)]), min([Version(1, 1)]))
