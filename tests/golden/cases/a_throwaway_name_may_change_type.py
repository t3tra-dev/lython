# A name rebound by a loop that NOTHING observes afterwards. The everyday
# spelling is the throwaway `_`, which a tuple unpack binds to a str and the
# next loop binds to an int:
#
#     value, _, count = part.partition("x")
#     for _ in range(int(count)): ...
#
# refused with "loop-carried local '_' is bound to str before the loop and to
# int inside it". The refusal's reason -- a loop that runs zero times leaves
# the earlier binding in place -- is a claim about a read AFTER the loop, and
# there is none: nothing here can see either binding.
#
# Golden and not an emit assertion because the values are the point. Not
# carrying a name changes which binding survives the loop, so a wrong answer
# looks like a working program that prints the pre-loop value. The last block
# is the control: the same shape WITH a read after the loop is still refused,
# and `tests/unit/EmitterTests.cpp` pins that half.
def expand(text: str) -> list[str]:
    out: list[str] = []
    for part in text.split(","):
        value, _, count = part.partition("x")
        for _ in range(int(count)):
            out.append(value)
    return out


print(expand("a x2"))
print(expand("ab x3,c x1"))
print(expand("z x0"))

_ = "a string"
for _ in range(3):
    pass
print("survived")

name = "a string"
for name in range(2):
    pass
print("also survived")

total = 0
for outer in range(3):
    step = "unused"
    for step in range(outer):
        total += 1
print(total)
