# WHAT: `list(filter(f, ...))` and `list(map(f, ...))` where `f` is a nested
# function reading the enclosing method's names, over `dir(cls)` of a class
# value and over a list.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the lazy iterator was a
# generator of its own at module level that named `f`, which names nothing
# there -- "unresolved runtime binding 'keep'" from the lowering -- and the
# result depends on the names the class value holds at run time.
class TC:
    def test_b(self) -> None:
        pass

    def test_a(self) -> None:
        pass


class Mine(TC):
    def test_c(self) -> None:
        pass


class Loader:
    prefix = 'test'

    def names(self, cls: type[TC]) -> list[str]:
        def keep(attrname: str) -> bool:
            return attrname.startswith(self.prefix)
        found = list(filter(keep, dir(cls)))
        found.sort()
        return found

    def shout(self, xs: list[str]) -> list[str]:
        def loud(x: str) -> str:
            return x.upper() + self.prefix
        return list(map(loud, xs))


print(Loader().names(Mine))
print(Loader().names(TC))
print(Loader().shout(["a", "b"]))
