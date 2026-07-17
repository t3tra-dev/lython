"""int <-> float conversions are exact / correctly rounded at any width:
float(big) rounds half-to-even (bit-identical to CPython, checked against
round-trip repr literals), int(float) reconstructs mantissa * 2**exponent
exactly beyond 2**63, and int / int rounds once like CPython."""

# float(int): exact below 2**53, round-half-even at the 53-bit boundary.
print(float(5))
print(float(-7))
print(float(2**53) == 9007199254740992.0)
print(float(2**53 + 1) == 9007199254740992.0)
print(float(2**53 + 2) == 9007199254740994.0)
print(float(2**53 + 3) == 9007199254740996.0)
print(float(-(2**53 + 1)) == -9007199254740992.0)
print(float(2**64 + 2048) == 1.8446744073709552e+19)
print(float(2**64 + 2049) == 1.8446744073709556e+19)
print(float(10**23) == 9.999999999999999e+22)
print(float(10**300 + 10**283) == 1.0000000000000001e+300)
print(float((2**53 + 1) * 2**100) == 1.1417981541647679e+46)
print(float(2**1023 + 2**970) == 8.98846567431158e+307)
# int(float): |x| >= 2**63 is exact (arbitrary-precision prints are exact).
print(int(2.5))
print(int(-2.5))
print(int(9.3e18))
print(int(-9.3e18))
print(int(1.8446744073709552e19))
print(int(1e30))
print(int(-1e300))
print(int(2.0 ** 1000.0) == 2 ** 1000)
print(int(5e-324))
# int / int: one correctly rounded step, exact at every scale.
print(7 / 2)
print(1 / 4)
print(-7 / 2)
print(10**400 / 10**399)
print(10**309 / 10**300)
print((2**52 + 1) / 2**52 == 1.0000000000000002)
print(10**23 / 1 == 9.999999999999999e+22)
print(1 / 10**23 == 1e-23)
print((2**53 + 1) / 2**53 == 1.0)  # tie rounds to even
print(1 / 2**1074 == 5e-324)
print(3 / 2**1075 == 1e-323)
print(2**1075 / 2**1075)
print(1 / 2**1075 == 0.0)
print(-1 / 2**1075 == 0.0)
print(0 / -5 == 0.0)
print((2**1024 - 2**971) / 1 == 1.7976931348623157e+308)
