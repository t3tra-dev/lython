# What: a builtin the EMITTER FOLDS, used INLINE rather than through a name.
# That is the shape: the fold makes the call run, and the inference walk knows
# nothing about it, so its callers widen the result to `object` -- which shows
# up only where a caller ASKS for the type, and never where the value is bound.
#
# `divmod` over floats used inline, without binding the pair first. The
# manifest's divmod is `[int, int] -> tuple[int, int]` and the emitter rewrites
# the float form into `(a // b, a % b)` -- so the call ran and the inference
# walk answered nothing, which its callers widen to `object`:
#
#     print(str(divmod(7.5, 2.5)))
#     # cannot pass concrete object builtins.tuple as builtins.object runtime
#     # input 0 of builtins.object.__str__
#
# while `t = divmod(7.5, 2.5)` then `str(t)` compiled, and the int form
# compiled inline. One question, two spellings: the walk now states the
# emitter's own rule, which its note gives as "divmod(x, y) and (x // y, x % y)
# are the same quotient and the same remainder".
#
# Running it is what pins the pair: the quotient is a float and the remainder
# is a float, and an int-typed answer would print without the `.0`.
def pair(a: float, b: float) -> str:
    return str(divmod(a, b))


def unpacked(a: float, b: float) -> str:
    q, r = divmod(a, b)
    return str(q) + " " + str(r)


def mixed(a: float, b: int) -> str:
    return str(divmod(a, b))


print("inline", pair(7.5, 2.5))
print("module scope", str(divmod(7.5, 2.5)))
print("unpacked", unpacked(7.5, 2.0))
print("mixed", mixed(7.5, 2))
print("ints still", str(divmod(7, 3)))

# ⭐ The same shape one builtin over: `sorted(x, reverse=True)` is a SUGAR
# rewrite in the emitter, so the walk had nothing for it and `str()` of it
# inline was the same message about `builtins.list`. `key=` does not change
# the answer -- the result holds the argument's elements, whatever the key
# ordered them by.
values: list[int] = [3, 1, 2]
words: list[str] = ["bb", "a", "ccc"]

print("sorted reverse", str(sorted(values, reverse=True)))
print("sorted key", str(sorted(words, key=len)))
print("sorted plain", str(sorted(values)))
