# What: round(int, ndigits) with ndigits < 0 returns a negative result without
# touching the shared objects small ints are (-5 through 256, as in CPython).
# It signed the product in place, and `round(-15, -1)` made every 20 in the
# program -20 -- a literal, a list element, an arithmetic result.
#
# Why run: the corruption is in the values later code reads at run time.
print(round(-15, -1), round(-25, -1), round(-149, -2), round(-151, -2))
b = [1, 2, 3]
b[::2] = [10, 20]
print(b)
x = 20
print(x, 10 + 10, 20 * 1, -(-20))
print(30, 100, 200)
