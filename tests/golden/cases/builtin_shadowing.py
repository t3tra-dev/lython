# Why this needs execution and not an emit assertion: the defect was that the
# builtin ran INSTEAD of the user's function and still exited 0, so what has to
# be pinned is the value each call produces. Emit-level tests pin the mechanism
# (EmitterTest.TopLevelDefOutranksBuiltinFastPath); only running pins the answer.
#
# One name per resolution layer that used to decide this differently:
#   len/abs/hash/ord    manifest builtin binding (renamed emitted symbol)
#   repr/round/sum      builtin fast path with no manifest binding
#   int/str             builtin class contract, not a free function
#   sorted              rewrite sugar with its own shadowing guard
# and both arities, because the winner used to depend on argument count.


def len(a: list[int]) -> int:
    return 99


def abs(a: int) -> int:
    return 98


def hash(a: int, b: int) -> int:
    return 97


def ord(a: str) -> int:
    return 96


def repr(a: int) -> str:
    return "user-repr"


def round(a: float) -> int:
    return 95


def sum(a: list[int]) -> int:
    return 94


def int(a: float) -> int:
    return 93


def str(a: int, b: int) -> int:
    return 92


def sorted(a: list[int]) -> int:
    return 91


print(len([1, 2, 3]))
print(abs(-7))
print(hash(1, 2))
print(ord("a"))
print(repr(1))
print(round(1.4))
print(sum([1, 2, 3]))
print(int(1.5))
print(str(1, 2))
print(sorted([2, 1]))


def recurse(n: int) -> int:
    # A shadowing def is still spelled by its own name inside its own body.
    if n == 0:
        return 0
    return len([1]) + recurse(n - 1)


print(recurse(2))


def local_shadow() -> int:
    # A local binding shadows the builtin as well; this arity used to match the
    # fast path, which had no shadowing gate.
    len = abs
    return len(-5)


print(local_shadow())
