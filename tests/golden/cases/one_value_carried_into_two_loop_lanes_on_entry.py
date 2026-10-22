# WHAT: a loop entered with two locals bound to the SAME freshly made value
# -- a str, a list, a str narrowed out of an Optional -- each rebound to one
# new value per trip and left by a `break` in the middle of the body, run
# for zero and several trips.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the refcount insertion
# folded the two header arguments into one, which then got two tokens from
# the back edge and one from the entry ("released or transferred more than
# once", re's _parse_flags); that each trip leaves both names holding live
# values is seen in what is read after the loop.
def get(text: str, i: int) -> str | None:
    if i < len(text):
        return text[i]
    return None


def until_colon(text: str) -> str:
    n = 0
    got = get(text, n)
    if got is None:
        return "empty"
    char = got
    while True:
        got = get(text, n)
        if got is None:
            return "unterminated after " + char
        char = got
        if char == ":":
            break
        n += 1
    return "%s %s %d" % (got, char, n)


def grow(limit: int) -> str:
    n = 0
    got = [n]
    both = got
    while True:
        got = [n] * n
        both = got
        if len(both) >= limit:
            break
        n += 1
    return "%r %r" % (got, both)


def word(i: int) -> str:
    return "w%d" % i


def last_word(limit: int) -> str:
    n = 0
    got = word(n)
    seen = got
    while n < 100:
        got = word(n)
        seen = got
        if n >= limit:
            break
        n += 1
    return got + seen


for text in [":", "ab:", "abc", ""]:
    print(until_colon(text))
for limit in [0, 1, 3]:
    print(grow(limit))
for limit in [0, 4]:
    print(last_word(limit))
