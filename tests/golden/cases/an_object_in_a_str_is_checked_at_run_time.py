# What: `x in s` with `x: object` and `s: str` searches when x holds a str and
# raises CPython's TypeError naming x's class when it does not.
# Why run: what x holds is a run-time fact.
def has(x: object, s: str) -> bool:
    return x in s


def lacks(x: object, s: str) -> bool:
    return x not in s


print(has("a", "abc"), has("z", "abc"), lacks("b", "abc"))
for v in [3, None, ["a"]]:
    try:
        print(has(v, "abc"))
    except TypeError as e:
        print("TypeError:", e)
