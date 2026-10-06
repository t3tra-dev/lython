# WHAT: complex arithmetic gives CPython 3.14's answers: abs is hypot (no
# overflow or underflow in the squares), / is Smith's division with C11
# Annex G recovery, a real operand combines with a complex directly (the
# sign of a zero part survives), ** takes small integer exponents by repeated
# multiplication, .real / .imag / conjugate() / == with a real read the
# parts, and truth is either part nonzero. Every value here is the same on
# an FMA and a non-FMA target.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defects were wrong
# NUMBERS -- inf for a finite modulus, nan for a quotient of 1, a raised
# ZeroDivisionError for a representable one -- produced by the runtime's
# arithmetic; only executing it shows the value.


def mixed(x: int) -> complex:
    z = 2j
    return z + x


z = complex(1.0, -0.0)
w = complex(2.0, 3.0)
x = 2.5
n = 3
print(abs(complex(1e200, 1e200)), abs(complex(3e-200, 4e-200)))
print(complex(1e300, 1e300) / complex(1e300, 1e300))
print(complex(1.0, 0.0) / complex(1e-200, 0.0))
print(z + 1.0, 1.0 + z, z * 2, 2.0 * z, z / 2.0, x - z)
print(mixed(1), n * w, w + x, w - n, n - w)
print(w ** 2, w ** n, w ** -3, z ** 2, 2 ** complex(0.0, 0.0))
print(w.real, w.imag, z.imag, w.conjugate())
print(z == 1.0, 1 == z, w != x, w == complex(2, 3), 2.0 == w)
print(bool(w), bool(0j), bool(complex(0.0, -0.0)), "yes" if w else "no")
print(complex(float("inf"), 1.0) * complex(1.0, 1.0))
print(complex(1.0, 1.0) / complex(float("inf"), 0.0))
try:
    print(complex(1.0, 1.0) / 0j)
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
try:
    print(w / 0.0)
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
try:
    print(0j ** -1)
except ZeroDivisionError as e:
    print("ZeroDivisionError:", e)
try:
    print(complex(1e300, 1e300) ** 5)
except OverflowError as e:
    print("OverflowError:", e)
try:
    print(abs(complex(1.7e308, 1.7e308)))
except OverflowError as e:
    print("OverflowError:", e)
