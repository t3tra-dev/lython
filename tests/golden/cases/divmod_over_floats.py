# What this pins: `divmod` when either operand is a float.
#
#     print(divmod(7.5, 2))
#     # static type !py.callable<[builtins.int, builtins.int], ...> is not
#     # callable with these arguments
#
# The manifest's divmod is typed [int, int], and the pair it would answer is
# one this compiler already computes: CPython defines divmod(x, y) and
# (x // y, x % y) together -- same quotient, same remainder -- so the float
# call is that pair.
#
# Why this must run: the negative cases are the answer. Python's floor division
# rounds toward negative infinity and its remainder takes the divisor's sign,
# so divmod(-7.5, 2) is (-4.0, 0.5) and divmod(7.5, -2) is (-4.0, -0.5) -- two
# results that a C-style truncating implementation gets wrong in both places
# and that no type can show.
#
# ⛔ The operands are bound to temporaries first, because the rewrite names each
# of them twice: `divmod(f(), 2)` must call f() once, and the counter here is
# what says it does.
#
# ⛔ int // int stays on the manifest's own divmod -- the rewrite is only for the
# float pair -- so the int answers are here to show it was not disturbed.
calls = 0


def once() -> float:
    global calls
    calls += 1
    return 7.5


print(divmod(7.5, 2), divmod(7, 2.0), divmod(7.5, 2.5))
print(divmod(-7.5, 2), divmod(7.5, -2), divmod(-7.5, -2))
print(divmod(0.0, 2), divmod(1.5, 0.5), divmod(-0.0, 2))
print(divmod(7, 3), divmod(-7, 3), divmod(7, -3))
print(divmod(once(), 2), "calls", calls)

try:
    print(divmod(1.0, 0))
except ZeroDivisionError as e:
    print("zero", e)

q, r = divmod(17.5, 5)
print(q, r, q * 5 + r == 17.5)
