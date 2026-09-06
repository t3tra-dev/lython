# What: `divmod` over floats used INLINE, without binding the pair first. The
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
