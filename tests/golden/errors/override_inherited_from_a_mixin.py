# B declares nothing, but its MRO resolves `v` to the mixin's, not to A's.
# Testing only what a subclass DECLARES missed this: the base's body was
# inlined for a receiver whose class resolves the method elsewhere.
class A:
    def v(self) -> int:
        return 1


class M:
    def v(self) -> int:
        return 2


class B(M, A):
    pass


a: A = B()
print(a.v())
