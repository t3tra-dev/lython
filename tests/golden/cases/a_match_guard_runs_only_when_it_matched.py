# What this pins: `if` guards on match cases -- when they run, and on which
# patterns they are allowed at all.
#
#     match x:
#         case 1 if note():   # note() was called for x == 2, where CPython
#             ...             # does not call it
#
#     match p:
#         case P(x=n) if n > 3:
#         # match class pattern requires a statically resolved class with
#         # capture or literal sub-patterns (no guards)
#
# The guard used to be ANDed with the pattern's condition in one block, which
# evaluates both whatever the subject is; a branch is what sequences them. The
# class, sequence and mapping arms refused a guard outright instead of placing
# it -- and the place is the same one CPython uses: after the captures are
# bound, so `case [a, b] if a < b` can see them, and after the element and key
# tests, so a subject of the wrong shape never reaches the guard.
#
# Why this must run: the whole defect is HOW MANY TIMES a function is called,
# which is a counter in a running program and nothing a type can show. Each
# guard here is a call that counts itself, and each block's count is the
# assertion.
class P:
    __match_args__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


calls: int = 0


def big(v: int) -> bool:
    global calls
    calls += 1
    return v > 1


for n in [1, 2]:
    match n:
        case 1 if big(n):
            print("one big")
        case 1:
            print("one")
        case _:
            print("other", n)
print("literal guards", calls)

calls = 0
for p in [P(1, 2), P(3, 4), P(0, 0)]:
    match p:
        case P(x=0):
            print("zero")
        case P(x=v) if big(v):
            print("class big", v)
        case P(x=v):
            print("class small", v)
print("class guards", calls)

calls = 0
for xs in [[1, 2], [3, 4], [5]]:
    match xs:
        case [a, b] if big(a):
            print("pair big", a, b)
        case [a, b]:
            print("pair", a, b)
        case _:
            print("not a pair")
print("sequence guards", calls)

calls = 0
for d in [{"k": 1}, {"k": 5}, {"j": 9}]:
    match d:
        case {"k": v} if big(v):
            print("map big", v)
        case {"k": v}:
            print("map small", v)
        case _:
            print("no k")
print("mapping guards", calls)

x = 5
match x:
    case int() if x > 3:
        print("int guard")
    case _:
        print("unreachable")
