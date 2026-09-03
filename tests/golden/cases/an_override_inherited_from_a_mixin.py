# B declares nothing, but its MRO resolves `v` to the mixin's, not to A's, and
# `tag` the same way. Testing only what a subclass DECLARES missed this once --
# the base's body was inlined for a receiver whose class resolves the method
# elsewhere -- and then, once that was a refusal, the dispatcher's own
# candidate scan still asked the narrow question and had nothing to dispatch
# to. Which base comes FIRST is the whole answer, so both orders are here.


class A:
    tag = "a"

    def v(self) -> int:
        return 1


class M:
    tag = "m"

    def v(self) -> int:
        return 2


class MixinFirst(M, A):
    pass


class BaseFirst(A, M):
    pass


def show(a: A) -> str:
    return a.tag + str(a.v())


for value in [A(), MixinFirst(), BaseFirst()]:
    print(show(value))
