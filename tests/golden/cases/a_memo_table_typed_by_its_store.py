# The scan that decides an empty container's element type binds what each
# statement leaves behind for the ones after it -- so that a seed written in
# terms of a local reads that local rather than `object`. It did that only for
# the suites it reached through a STATEMENT, and not for the top-level ones it
# is handed, which is where a memo table's store lives:
#
#     CACHE = {}
#     def fib(n: int) -> int:
#         ...
#         value = fib(n - 1) + fib(n - 2)
#         CACHE[n] = value          <- `value` was unbound here
#         return value
#     print(CACHE[10] + 1)
#     # builtins.object does not provide manifest method '__add__'
#
# Why execution: the table's value type decides what may be read back out of
# it, so the program has to DECODE what it memoised -- and the memo has to
# actually hold, which only running it shows.
#
# ⭐ ONE ANSWER, ASKED FROM BOTH WALKS. The two places that walk a suite -- the
# generic recursion into a statement's own suites, and the forward scan over
# the ones handed in -- ask the same question now.
#
# ⛔ Still bound AFTER the statement is scanned, so an assignment does not see
# itself: `out = out + [1]` reads `out` at the type being decided, which is the
# reason that shape is skipped rather than counted.


CACHE = {}
CALLS = []


def fib(n: int) -> int:
    if n < 2:
        return n
    if n in CACHE:
        return CACHE[n]
    CALLS.append(n)
    value = fib(n - 1) + fib(n - 2)
    CACHE[n] = value
    return value


def longest(words: "list[str]") -> str:
    best = ""
    table = {}
    for w in words:
        size = len(w)
        table[w] = size
        if size > len(best):
            best = w
    return best + ":" + str(table[best])


print(fib(20))
print(len(CACHE), CACHE[10] + 1)
print(len(CALLS), CALLS[0] + 1)
print(longest(["ant", "beetle", "cow"]))
