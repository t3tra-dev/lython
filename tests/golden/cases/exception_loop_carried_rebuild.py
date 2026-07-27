# A loop-carried exception entity rebuilt on every iteration.
#
# `LyException_Init` consumes its receiver and hands it back (transfer_args
# = [0, 3], owned_results = [0], result_evidence = "receiver"), so the
# construction inside the loop body moves a token to a new name and leaves the
# pre-transfer name behind. Carried across the back edge, that name used to stay
# in the ownership walk's stale set past the very call that rebinds it, and every
# spelling below was refused with "released through a value already consumed by
# an ownership transfer" -- ordinary Python, and CPython's answer is the last
# assignment.
#
# Every loop here COMPLETES and rebinds on every iteration, which is the part
# that matters: a loop that ends by raising never reaches the release that the
# rebind makes due, and a loop that does not mutate never creates the second
# name. `big` runs the balance 500 times rather than once, so an off-by-one in
# either direction shows as output rather than as a coincidence.
class Err(Exception):
    pass


def while_loop(n: int) -> str:
    cur: BaseException = ValueError("seed")
    i = 0
    while i < n:
        cur = ValueError("attempt" + str(i))
        i += 1
    return str(cur)


def for_loop(n: int) -> str:
    cur: BaseException = ValueError("seed")
    for i in range(n):
        cur = ValueError("attempt" + str(i))
    return str(cur)


def user_class(n: int) -> str:
    cur: BaseException = Err("seed")
    i = 0
    while i < n:
        cur = Err("attempt" + str(i))
        i += 1
    return str(cur)


def nested(n: int) -> str:
    cur: BaseException = ValueError("seed")
    i = 0
    while i < n:
        j = 0
        while j < n:
            cur = Err("inner" + str(i) + str(j))
            j += 1
        i += 1
    return str(cur)


def two_carried(n: int) -> str:
    a: BaseException = ValueError("a-seed")
    b: BaseException = Err("b-seed")
    i = 0
    while i < n:
        a = ValueError("a" + str(i))
        b = Err("b" + str(i))
        i += 1
    return str(a) + "/" + str(b)


def used_in_body(n: int) -> str:
    cur: BaseException = ValueError("seed")
    seen = 0
    i = 0
    while i < n:
        cur = ValueError("attempt" + str(i))
        seen += len(str(cur))
        i += 1
    return str(cur) + ":" + str(seen)


def big(n: int) -> str:
    cur: BaseException = ValueError("seed")
    i = 0
    while i < n:
        cur = ValueError("x")
        i += 1
    return str(cur)


# The negative control for the same loop shape: an ordinary class whose
# constructor declares no transfer. It worked before and must go on working, or
# the change reached further than the transferred-receiver contracts.
class Plain:
    def __init__(self, s: str) -> None:
        self.s = s


def plain_loop(n: int) -> str:
    cur: Plain = Plain("seed")
    i = 0
    while i < n:
        cur = Plain("attempt" + str(i))
        i += 1
    return cur.s


print(while_loop(0))
print(while_loop(1))
print(while_loop(3))
print(for_loop(3))
print(user_class(3))
print(nested(2))
print(two_carried(3))
print(used_in_body(3))
print(big(500))
print(plain_loop(3))

# Module scope carries its own loop-carried local through a different emitter
# path than a function body does, and it was refused too.
top: BaseException = ValueError("top-seed")
k = 0
while k < 3:
    top = Err("top" + str(k))
    k += 1
print(str(top))
